set -ex



# example training scripts for AAAI-21
# Split then Refine: Stacked Attention-guided ResUNets for Blind Single Image Visible Watermark Removal


CUDA_VISIBLE_DEVICES=0 python /data/home/yb87432/s2am/main.py  --epochs 100\
 --schedule 100\
 --lr 1e-3\
 -c eval/10kgray/1e3_bs4_256_hybrid_ssim_vgg\
 --arch vvv4n\
 --sltype vggx\
 --style-loss 0.025\
 --ssim-loss 0.15\
 --masked True\
 --loss-type hybrid\
 --limited-dataset 1\
 --machine vx\
 --input-size 256\
 --train-batch 4\
 --test-batch 1\
 --base-dir $HOME/watermark/10kgray/\
 --data _images


python main.py  --epochs 100 --schedule 100 --lr 1e-3 -c ckpt --arch vvv4n --sltype vggx --style-loss 0.025 --ssim-loss 0.15 --masked True --loss-type hybrid --limited-dataset 1 --machine vx --input-size 256 --train-batch 4 --test-batch 1 --base-dir ../tmp/split_and_refine_ds/ --data _images



# example training scripts for TIP-20
# Improving the Harmony of the Composite Image by Spatial-Separated Attention Module
# * in the original version, the res = False
# suitable for the iHarmony4 dataset.

python /data/home/yb87432/mypaper/s2am/main.py  --epochs 200\
 --schedule 150\
 --lr 1e-3\
 -c checkpoint/normal_rasc_HAdobe5k_res \
 --arch rascv2\
 --style-loss 0\
 --ssim-loss 0\
 --limited-dataset 0\
 --res True\
 --machine s2am\
 --input-size 256\
 --train-batch 16\
 --test-batch 1\
 --base-dir $HOME/Datasets/\
 --data HAdobe5k





total Dataset of train is :  14400
total Dataset of val is :  2015
==> creating model 
/opt/conda/lib/python3.10/site-packages/torch/functional.py:513: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at /usr/local/src/pytorch/aten/src/ATen/native/TensorShape.cpp:3609.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
==> creating model [Finish]
/opt/conda/lib/python3.10/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/opt/conda/lib/python3.10/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=VGG16_Weights.IMAGENET1K_V1`. You can also use `weights=VGG16_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to /root/.cache/torch/hub/checkpoints/vgg16-397923af.pth
100%|█████████████████████████████████████████| 528M/528M [00:02<00:00, 187MB/s]
==> Total params: 32.61M
==> Total devices: 2
==> Current Checkpoint: ckpt_vx__vvv4n
Traceback (most recent call last):
  File "/kaggle/working/deep-blind-watermark-removal/main.py", line 89, in <module>
    main(args)
  File "/kaggle/working/deep-blind-watermark-removal/main.py", line 31, in main
    Machine = machines.__dict__[args.machine](datasets=data_loaders, args=args)
  File "/kaggle/working/deep-blind-watermark-removal/scripts/machines/__init__.py", line 13, in vx
    return VX(**kwargs)
  File "/kaggle/working/deep-blind-watermark-removal/scripts/machines/VX.py", line 92, in __init__
    self.model.set_optimizers()
  File "/opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1729, in __getattr__
    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
AttributeError: 'DataParallel' object has no attribute 'set_optimizers'