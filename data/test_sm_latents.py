import os,tempfile,json,copy,unittest
from pathlib import Path
from dataclasses import asdict
import torch
from omegaconf import OmegaConf
from hydra import compose,initialize_config_dir
from modules.submodules.MelCausalVAE.dicodec.modules.sm_quantizer.configs import Config,from_dict
from modules.submodules.MelCausalVAE.dicodec.modules.sm_quantizer.model import SMQuantizer
from data.sm_latents import SMLatentCollator
from util import build_tokenizer
from modules.builder import build_model


class SMLatentTests(unittest.TestCase):
    def test_five_training_paths(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
         root=Path(directory)
         (root/'config.json').write_text(json.dumps({'latent_dim':64}))
         s=from_dict(Config,{'model':{'language_modeling':False},'data':{'input':'z_sem','target':'z_sem'}})
         sm=SMQuantizer(s.model)
         ck=root/'last.pt';torch.save({'config':asdict(s),'model':sm.state_dict()},ck)
         os.environ['SLURM_TMPDIR']=directory
         with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / 'configs'),version_base=None):
          for name in ('asr','lm','asr_and_lm','recon','continuous_only'):
           c=OmegaConf.to_container(compose(config_name='main',overrides=[f'settings=slm/{name}']),resolve=True)
           c['vae_checkpoint']=str(root)
           c['training']['sm_quantizer_checkpoint']=str(ck) if name!='continuous_only' else None
           c['backbone'].update(num_layers=1,num_heads=2,n_kv_heads=1,hidden_dim=32,ffn_dim=64)
           c['diffusion_head'].update(net_dim=32,net_heads=2,net_depth=2,backbone_dim=32)
           c['continuous_adapter'].update(hidden_dim=32,num_layers=1,out_dim=32)
           tok=build_tokenizer(c)
           model=build_model(c,tok)
           collator=SMLatentCollator(tok,c['training']['sm_quantizer_checkpoint'],'cpu',discrete=c['training']['discrete'])
           items=[{'z':torch.randn(n,64),'attributes':{'z_sem':torch.randn(n,64)},'transcript':text} for n,text in ((5,'hi'),(3,'a'))]
           b=collator(items)
           torch.testing.assert_close(b['continuous_sequence'][0,:5],items[0]['z'])
           assert b['discrete_sequence'][0,:2].tolist()==tok.encode_text('hi')
           o=model(discrete_sequence=b['discrete_sequence'],attention_mask=b['attention_mask'],continuous_sequence=b['continuous_sequence'],audio_padding_mask=b['audio_padding_mask'])
           loss=o.diffusion_loss
           if c['training']['discrete']:
            loss=loss+torch.nn.functional.cross_entropy(o.token_logits.reshape(-1,tok.discrete_token_vocab_size+1),b['target_tokens'].reshape(-1),ignore_index=-100)
           loss.backward()
           assert torch.isfinite(loss)
           if collator.quantizer:
            assert all(p.grad is None for p in collator.quantizer.parameters())
           print(name,'PASS',float(loss.detach()),flush=True)

    def test_frozen_backends_and_parquet(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from data.sm_latents import FrozenSMTokenizer, build_latent_dataset
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for kind in ('vq_ema', 'fsq', 'bsq'):
                config = from_dict(Config, {'model': {'language_modeling': False, 'quantizer': {'type': kind}},
                                           'data': {'input': 'z_sem', 'target': 'z_sem'}})
                model = SMQuantizer(config.model).eval()
                path = root / f'{kind}.pt'
                torch.save({'config': asdict(config), 'model': model.state_dict()}, path)
                frozen = FrozenSMTokenizer(str(path))
                z = torch.randn(2, 5, 64)
                valid = torch.tensor([[True]*5, [True, True, False, False, False]])
                before = copy.deepcopy(frozen.state_dict())
                expected = model.encode(z, valid).indices
                z[~valid] = float('nan')
                torch.testing.assert_close(frozen(z, valid), expected)
                for key, value in before.items():
                    torch.testing.assert_close(frozen.state_dict()[key], value)
                self.assertFalse(any(p.requires_grad for p in frozen.parameters()))
                self.assertFalse(any(key.startswith(('asr_head.', 'decoder.', 'reconstruction_head.')) for key in frozen.state_dict()))
            for partition, text in [('train_clean_100', 'hello'), ('train_clean_360', 'world')]:
                dest = root / partition
                dest.mkdir()
                pq.write_table(pa.Table.from_pylist([{'z': [[1.]*64]*3, 'attributes': {'z_sem': [[2.]*64]*3},
                                                       'transcript': text}]), dest / 'data.parquet')
            train, val = build_latent_dataset({'latent_dataset_path': str(root),
                'dataset_partitions': ['train_clean_100','train_clean_360'], 'discrete': True,
                'latent_cache_dir': str(root/'cache')})
            self.assertEqual(len(train), 2)
            self.assertEqual(len(val), 0)
            self.assertEqual(train[1]['transcript'], 'world')

if __name__ == '__main__':
    unittest.main()
