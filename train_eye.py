from eyeData import EyeDataProvider
from tf_unet import unet
import platform


# nx,ny = 512, 1024

data_path = "/Users/miao/Downloads/eyedata/Edema_?????*set/original_images/*/*.bmp" if platform.system()=="Darwin" \
    else "/root/eyedata/Edema_?????*set/original_images/*/*.bmp"

eye_data_provider = EyeDataProvider(data_path, n_class=4)

net = unet.Unet(channels=eye_data_provider.channels,
                n_class=eye_data_provider.n_class,
                layers=3,
                features_root=64,
                cost_kwargs=dict(regularizer=0.001),
                )

trainer = unet.Trainer(net, optimizer="momentum", opt_kwargs=dict(momentum=0.2))
path = trainer.train(eye_data_provider, "./unet_trained_eye_data",
                     training_iters=32,
                     epochs=1,
                     dropout=0.5,
                     display_step=1)

