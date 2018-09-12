#/bin/bash

docker run -d \
    --name tf_unet \
    -v $PWD:/root/tf_unet \
    -w /root/tf_unet \
    --device=/dev/nvidiactl --device=/dev/nvidia-uvm --device=/dev/nvidia2 --device=/dev/fuse \
    --volume /var/drivers/nvidia/current:/usr/local/nvidia:ro
    mzmssg/eye sleep infinity

docker exec -it tf_unet /bin/bash 

