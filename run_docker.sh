#/bin/bash

GPU_ID = 2 # choose from 0,1,2,3

docker run -d \
    --name tf_unet \
    --net=host \
    -v $PWD:/root/tf_unet \
    -w /root/tf_unet \
    --device=/dev/nvidiactl --device=/dev/nvidia-uvm --device=/dev/nvidia$GPU_ID --device=/dev/fuse \
    --volume /var/drivers/nvidia/current:/usr/local/nvidia:ro \
    mzmssg/eye sleep infinity

docker exec -it tf_unet /bin/bash 

