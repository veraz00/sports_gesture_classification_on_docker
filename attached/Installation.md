# How to install docker File 

1. Install client and server docker
```
sudo apt-get purge docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin docker-ce-rootless-extras
sudo rm -rf /var/lib/docker
sudo rm -rf /var/lib/containerd

sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo docker version 
````

- If installation was successful, it would return 
```
$ docker version 
Client: Docker Engine - Community
 Version:           23.0.0
 API version:       1.42
 Go version:        go1.19.5
 Git commit:        e92dd87
 Built:             Wed Feb  1 17:49:08 2023
 OS/Arch:           linux/amd64
 Context:           default

Server: Docker Engine - Community
 Engine:
  Version:          23.0.0
  API version:      1.42 (minimum version 1.12)
  Go version:       go1.19.5
  Git commit:       d7573ab
  Built:            Wed Feb  1 17:49:08 2023
  OS/Arch:          linux/amd64
  Experimental:     false
 containerd:
  Version:          1.6.16
  GitCommit:        31aa4358a36870b21a992d3ad2bef29e1d693bec
 nvidia:
  Version:          1.1.4
  GitCommit:        v1.1.4-0-g5fd4c4d
 docker-init:
  Version:          0.19.0
  GitCommit:        de40ad0
```

2. Install nvidia-container 
refer: [installation on ubuntu](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

- If installation was successful, run 
```
sudo docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
{nvidia-smi local version}
```

# Error 
1. `Docker with GPU: "Failed to initialize NVML: Unknown Error"`
- Solution: https://bbs.archlinux.org/viewtopic.php?id=266915

2. Error 
```
docker: Error response from daemon: failed to create shim task: OCI runtime create failed: runc create failed: unable to start container process: error during container init: error running hook #0: error running hook: exit status 1, stdout: , stderr: Auto-detected mode as 'legacy'
nvidia-container-cli: initialization error: load library failed: libnvidia-ml.so.1: cannot open shared object file: no such file or directory: unknown.
ERRO[0000] error waiting for container: context canceled 

```
- Reason
    - (1) you are not running the nvidia-persistenced daemon
    - (2) your GPUs are not in persistence mode.

- Solution: Set nvidia-persistenced daemon on  (refer: https:/dev.to/bybatkhuu/install-nvidia-gpu-driver-on-linux-ubuntudebian-4ei6)

```
# Install git to clone:
sudo apt-get install -y ssh git

# Download nvidia-persistenced source code from github:
git clone https://github.com/NVIDIA/nvidia-persistenced.git

# Install nvidia-persistenced service daemon:
cd nvidia-persistenced/init
sudo ./install.sh

# Remove downloaded files:
cd ../.. && rm -rf nvidia-persistenced

# Check nvidia-persistence mode is ON:
nvidia-smi
# Or check nvidia-persistenced.service is running:
systemctl status nvidia-persistenced.service

# Disable the nvidia-persistenced service daemon
# Stop and disable nvidia-persistenced service daemon:
sudo systemctl stop nvidia-persistenced.service
sudo systemctl disable nvidia-persistenced.service
```

- Persistence mode can be set using nvidia-smi or programmaticaly via the NVML API. To enable persistence mode using nvidia-smi (as root): nvidia-smi -i <target gpu> -pm ENABLED 


3. Issues 

```
linlin@linlin:~$ sudo apt-get update
E: Conflicting values set for option Signed-By regarding source https://nvidia.github.io/libnvidia-container/stable/ubuntu18.04/amd64/ /: /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg != 
E: The list of sources could not be read.

```

- solution: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/troubleshooting.html#conflicting-values-set-for-option-signed-by-error-when-running-apt-update



4. add user permission 
```
docker: permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Post "http://%2Fvar%2Frun%2Fdocker.sock/v1.24/containers/create": dial unix /var/run/docker.sock: connect: permission denied.
```
- solution: `$chmod 777 /var/run/docker.sock`