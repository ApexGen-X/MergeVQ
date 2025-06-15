import torch
import torch.nn.functional as F
import lightning as L

from main import instantiate_from_config
from contextlib import contextmanager
from collections import OrderedDict

from taming.modules.diffusionmodules.model_vqgan import Encoder, Decoder
from taming.modules.diffusionmodules.model_mergevq import MergeVQEncoder, MergeVQDecoder, SourceRecovery, token_unmerge
from taming.modules.vqvae.lookup_free_quantize import LFQ
from taming.modules.vqvae.quantize import EMAVectorQuantizer
from taming.modules.scheduler.lr_scheduler import Scheduler_LinearWarmup, Scheduler_LinearWarmup_CosineDecay
from taming.modules.ema import LitEma


class VQModel(L.LightningModule):
    def __init__(self,
                ddconfig,
                lossconfig,
                k2lconfig = None,
                ## Quantize Related
                n_embed = 262144,
                embed_dim = 18,
                sample_minimization_weight = 1.0,
                batch_maximization_weight = 1.0,
                vq_method = "lfq",
                ckpt_path = None,
                ignore_keys = [],
                image_key = "image",
                colorize_nlabels = None,
                monitor = None,
                learning_rate = None,
                optimizer = "Adam",
                opt_betas = (0.5, 0.9),
                resume_lr = None,
                pretrain_decoder = None,
                ### scheduler config
                warmup_epochs = 1.0, #warmup epochs
                scheduler_type = "linear-warmup_cosine-decay",
                min_learning_rate = 0,
                use_ema = False,
                token_factorization = False,
                stage = None,
                lr_drop_epoch = None,
                lr_drop_rate = 0.1,
                factorized_bits = [9, 9],
                gradient_clip_val = None,
                accumulate_grad_batches = 1
                ):
        super().__init__()
        self.image_key = image_key
        if k2lconfig is None:
            self.encoder = Encoder(**ddconfig)
            self.decoder = Decoder(**ddconfig)
            self.recovery = None
        else:
            self.encoder = MergeVQEncoder(**ddconfig)
            if k2lconfig.get('cnn_decoder', False):
                self.decoder = Decoder(**ddconfig)
            else:
                self.decoder = MergeVQDecoder(**ddconfig)
            self.recovery = SourceRecovery(**k2lconfig)

        if pretrain_decoder is not None:
            decoder_ckpt = torch.load(pretrain_decoder)
            self.decoder.load_state_dict(decoder_ckpt['state_dict'], strict=True)

        self.loss = instantiate_from_config(lossconfig)
        self.quantize = LFQ(dim=embed_dim, codebook_size=n_embed, 
                            sample_minimization_weight=sample_minimization_weight, 
                            batch_maximization_weight=batch_maximization_weight, 
                            token_factorization=token_factorization, factorized_bits=factorized_bits)
        if vq_method == 'emavq':
            self.quantize = EMAVectorQuantizer(n_embed=n_embed, codebook_dim=embed_dim, beta=1.0)

        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        self.use_ema = use_ema
        if self.use_ema and stage is None: #no need to construct EMA when training Transformer
            self.model_ema = LitEma(self)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, stage=stage)
        self.resume_lr = resume_lr
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.opt_betas = opt_betas
        self.lr_drop_epoch = lr_drop_epoch
        self.lr_drop_rate = lr_drop_rate
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.min_learning_rate = min_learning_rate

        self.automatic_optimization = False  # manual optimization
        self.strict_loading = False
        self.gradient_clip_val = gradient_clip_val
        self.accumulate_grad_batches = accumulate_grad_batches

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def load_state_dict(self, *args, strict=False):
        """
        Resume not strict loading
        """
        return super().load_state_dict(*args, strict=strict)

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        '''
        filter out the non-used keys
        '''
        return {k: v for k, v in super().state_dict(*args, destination, prefix, keep_vars).items() \
            if ("inception_model" not in k and "lpips_vgg" not in k and "lpips_alex" not in k)}
        
    def init_from_ckpt(self, path, ignore_keys=list(), stage="transformer"):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        ema_mapping = {}
        new_params = OrderedDict()
        if stage == "transformer": ### directly use ema encoder and decoder parameter
            if self.use_ema:
                for k, v in sd.items(): 
                    if "encoder" in k:
                        if "model_ema" in k:
                            k = k.replace("model_ema.", "") #load EMA Encoder or Decoder
                            new_k = ema_mapping[k]
                            new_params[new_k] = v   
                        s_name = k.replace('.', '')
                        ema_mapping.update({s_name: k})
                        continue
                    if "decoder" in k:
                        if "model_ema" in k:
                            k = k.replace("model_ema.", "") #load EMA Encoder or Decoder
                            new_k = ema_mapping[k]
                            new_params[new_k] = v 
                        s_name = k.replace(".", "")
                        ema_mapping.update({s_name: k})
                        continue 
            else: #also only load the Generator
                for k, v in sd.items():
                    if "encoder" in k:
                        new_params[k] = v
                    elif "decoder" in k:
                        new_params[k] = v                  
        missing_keys, unexpected_keys = self.load_state_dict(new_params, strict=False) #first stage
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        (quant, emb_loss, info), loss_breakdown = self.quantize(h, return_loss_breakdown=True)

        return quant, emb_loss, info, loss_breakdown

    def decode(self, quant):
        if getattr(self.encoder.attn, "_tome_info", None) is None:  # default lfq
            dec = self.decoder(quant)
        else:
            # MergeVQ with token recovery
            source = self.encoder.attn._tome_info['source']
            if self.encoder.attn._tome_info['class_token'] and source is not None:
                source = source[:, 1:, 1:]
            quant = torch.flatten(quant, start_dim=2)
            quant = quant.permute(0, 2, 1)

            if self.recovery is not None:
                rec_tokens = self.recovery(quant, source)
            else:
                rec_tokens = token_unmerge(quant, source)
            rec_tokens = rec_tokens.permute(0, 2, 1)
            # print('rec', rec_tokens.shape)
            B, D, L = rec_tokens.shape
            rec_tokens = rec_tokens.reshape(B, D, int(L**0.5), -1)

            dec = self.decoder(rec_tokens)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _, loss_break = self.encode(input)
        dec = self.decode(quant)
        return dec, diff, loss_break

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.contiguous().float()
        return x

    def on_train_start(self):
        """change lr after resuming"""
        if self.resume_lr is not None:
            opt_gen, opt_disc = self.optimizers()
            for opt_gen_param_group, opt_disc_param_group in zip(opt_gen.param_groups, opt_disc.param_groups):
                opt_gen_param_group["lr"] = self.resume_lr
                opt_disc_param_group["lr"] = self.resume_lr

    def update_encoder_merge_ratio(self, batch_idx):
        """sampling and update merge ratio"""
        update_r = self.encoder.sampling_r_candidate(rand=(batch_idx % 100) * 0.01)

    # fix mulitple optimizer bug
    # refer to https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        self.update_encoder_merge_ratio(batch_idx)
        xrec, eloss,  loss_break = self.forward(x)

        opt_gen, opt_disc = self.optimizers()
        # scheduler_gen, scheduler_disc = self.lr_schedulers()

        ####################
        # fix global step bug
        # refer to https://github.com/Lightning-AI/pytorch-lightning/issues/17958
        opt_disc._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")
        opt_disc._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step")
        # opt_gen._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")
        # opt_gen._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step")
        ####################

        # optimize generator
        aeloss, log_dict_ae = self.loss(eloss, loss_break, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        if self.accumulate_grad_batches > 1:
            aeloss = aeloss / self.accumulate_grad_batches            
            self.manual_backward(aeloss)
            # accumulate gradients of N batches
            if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                if self.gradient_clip_val is not None:  # clip gradients
                    self.clip_gradients(opt_gen, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm="norm")
                opt_gen.step()
                opt_gen.zero_grad()  # clear grads at last
        else:
            opt_gen.zero_grad()
            self.manual_backward(aeloss)
            if self.gradient_clip_val is not None:  # clip gradients
                self.clip_gradients(opt_gen, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm="norm")
            opt_gen.step()
            # scheduler_gen.step()

        # optimize discriminator
        discloss, log_dict_disc = self.loss(eloss, loss_break, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
        if self.accumulate_grad_batches > 1:
            discloss = discloss / self.accumulate_grad_batches
            self.manual_backward(discloss)
            # accumulate gradients of N batches
            if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                if self.gradient_clip_val is not None:  # clip gradients
                    self.clip_gradients(opt_disc, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm="norm")
                opt_disc.step()
                opt_disc.zero_grad()
        else:
            opt_disc.zero_grad()
            self.manual_backward(discloss)
            if self.gradient_clip_val is not None:  # clip gradients
                self.clip_gradients(opt_disc, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm="norm")
            opt_disc.step()
            # scheduler_disc.step()

        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
    
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def on_train_epoch_end(self):
        ### update lr
        self.lr_annealing()
        torch.cuda.empty_cache()

    def lr_annealing(self):
        """
        Perform Lr decay
        """
        if self.lr_drop_epoch is not None:
            current_epoch = self.trainer.current_epoch
            if (current_epoch + 1) in self.lr_drop_epoch:
                opt_gen, opt_disc = self.optimizers()
                for opt_gen_param_group, opt_disc_param_group in zip(opt_gen.param_groups, opt_disc.param_groups):
                    opt_gen_param_group["lr"] = opt_gen_param_group["lr"] * self.lr_drop_rate
                    opt_disc_param_group["lr"] = opt_disc_param_group["lr"] * self.lr_drop_rate

    def validation_step(self, batch, batch_idx): 
        if self.use_ema:
            with self.ema_scope():
                log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        else:
            log_dict = self._validation_step(batch, batch_idx)

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        quant, eloss, indices, loss_break = self.encode(x)
        x_rec = self.decode(quant).clamp(-1, 1)
        aeloss, log_dict_ae = self.loss(eloss, loss_break, x, x_rec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val"+ suffix)

        discloss, log_dict_disc = self.loss(eloss, loss_break, x, x_rec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val" + suffix)
    
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        optimizer = torch.optim.AdamW if self.optimizer == 'AdamW' else torch.optim.Adam
        self.opt_betas = (self.opt_betas[0], self.opt_betas[1])
        opt_gen = optimizer(list(self.encoder.parameters())+
                            list(self.decoder.parameters())+
                            list(self.quantize.parameters()),
                            lr=lr, betas=self.opt_betas)
        opt_disc = optimizer(self.loss.discriminator.parameters(),
                             lr=lr, betas=self.opt_betas)

        if self.trainer.is_global_zero:
            print("step_per_epoch: {}".format(len(self.trainer.datamodule._train_dataloader()) // self.trainer.world_size))

        step_per_epoch  = len(self.trainer.datamodule._train_dataloader()) // self.trainer.world_size
        warmup_steps = step_per_epoch * self.warmup_epochs
        training_steps = step_per_epoch * self.trainer.max_epochs

        if self.scheduler_type == "None":
            return ({"optimizer": opt_gen}, {"optimizer": opt_disc})

        if self.scheduler_type == "linear-warmup":
            scheduler_ae = torch.optim.lr_scheduler.LambdaLR(opt_gen, Scheduler_LinearWarmup(warmup_steps))
            scheduler_disc = torch.optim.lr_scheduler.LambdaLR(opt_disc, Scheduler_LinearWarmup(warmup_steps))

        elif self.scheduler_type == "linear-warmup_cosine-decay":
            multipler_min = self.min_learning_rate / self.learning_rate
            scheduler_ae = torch.optim.lr_scheduler.LambdaLR(
                opt_gen, Scheduler_LinearWarmup_CosineDecay(warmup_steps=warmup_steps, max_steps=training_steps, multipler_min=multipler_min))
            scheduler_disc = torch.optim.lr_scheduler.LambdaLR(
                opt_disc, Scheduler_LinearWarmup_CosineDecay(warmup_steps=warmup_steps, max_steps=training_steps, multipler_min=multipler_min))
        else:
            raise NotImplementedError()
        return {"optimizer": opt_gen, "lr_scheduler": scheduler_ae}, {"optimizer": opt_disc, "lr_scheduler": scheduler_disc}

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


if __name__ == "__main__":

    ddconfig = dict(
        double_z = False,
        z_channels = 18,
        resolution = 256,
        in_channels = 3,
        out_ch = 3,
        ch = 128,
        ch_mult = [1,1,2,2,4],  # num_down = len(ch_mult)-1
        num_res_blocks = 2,
        num_att_blocks = 12,
        num_heads = 8,
        merge_ratio = 10,
        merge_num = 112,
    )
    k2lconfig = dict(
        num_merge_tokens = 64,
        num_rec_tokens = 256,
        embed_dim = 18,
        num_heads = 3,
        num_layers = 12,
        learn_source = False,
        cnn_decoder = False,
    )
    lossconfig = dict(
        target='taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator',
        params={
            "disc_conditional": False,
            "disc_in_channels": 3,
            "disc_start": 0,
            "disc_weight": 0.8,
            "gen_loss_weight": 0.1,
            "lecam_loss_weight": 0.05,
            "codebook_weight": 0.1,
            "commit_weight": 0.25,
            "codebook_enlarge_ratio": 0,
            "codebook_enlarge_steps": 2000,
        }
    )

    #class_path =  'taming.models.lfqgan.VQModel'
    n_embed = 262144
    embed_dim = 18
    learning_rate = 1e-4
    sample_minimization_weight = 1.0
    batch_maximization_weight = 1.0

    model = VQModel(ddconfig,
            lossconfig,
            k2lconfig,
            ## Quantize Related
            n_embed,
            embed_dim,
            sample_minimization_weight,
            batch_maximization_weight,
            vq_method = 'lfq',
            ckpt_path = None,
            ignore_keys = [],
            image_key = "image",
            colorize_nlabels = None,
            monitor = None,
            learning_rate = None,
            resume_lr = None,
            ### scheduler config
            warmup_epochs = 1.0, #warmup epochs
            scheduler_type = "linear-warmup_cosine-decay",
            min_learning_rate = 0,
            use_ema = False,
            token_factorization = False,
            stage = None,
            lr_drop_epoch = None,
            lr_drop_rate = 0.1,
            factorized_bits = [9, 9])

    x = torch.rand(1, 3, 256, 256)
    dec, _, _ = model(x)
    print(dec.shape)
