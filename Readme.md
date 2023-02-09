https://www.youtube.com/watch?v=Kzrfw-tAZew
https://github.com/abhishekkrthakur/melanoma-deep-learning



```
git clone --branch 22.06-dev https://github.com/NVIDIA/apex.git
cd apex
python setup.py install
```



## Error 
1. url perssion
- Error: `urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: certificate has expired (_ssl.c:1131)>`
- Solution
```
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```
2. get error: 
```
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].
```

- solution: reinstall docker, refer: [install docker on ubuntu](intallation.md)




