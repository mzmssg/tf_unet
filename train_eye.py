from eyeData import EyeDataProvider
from tf_unet import unet
import platform


# nx,ny = 512, 1024

train_data_path = "/Users/miao/Downloads/eyedata/Edema_trainset/original_images/*/*.bmp" if platform.system()=="Darwin" \
    else "/root/eyedata/Edema_trainset/original_images/*/*.bmp"

val_data_path = "/Users/miao/Downloads/eyedata/Edema_validationset/original_images/*/*.bmp" if platform.system()=="Darwin" \
    else "/root/eyedata/Edema_validationset/original_images/*/*.bmp"



train_data_provider = EyeDataProvider(train_data_path, n_class=4)
val_data_provider = EyeDataProvider(val_data_path, n_class=4)

net = unet.Unet(channels=train_data_provider.channels,
                n_class=train_data_provider.n_class,
                layers=3,
                features_root=64,
                cost_kwargs=dict(regularizer=0.001),
                )

batch_size = 16
training_iters = int(train_data_provider.data_num/batch_size)
epochs = 50


trainer = unet.Trainer(net, batch_size=batch_size, optimizer="momentum", opt_kwargs=dict(momentum=0.2))
path = trainer.train(train_data_provider, "./unet_trained_eye_data", 
                     val_data_provider=val_data_provider,                   
                     training_iters=training_iters,
                     epochs=epochs,
                     dropout=0.5,
                     display_step=1)

