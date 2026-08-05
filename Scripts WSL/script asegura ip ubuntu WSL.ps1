$wslIp = (wsl -d Ubuntu -e hostname -I).Trim()
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=3000 connectaddress=$wslIp connectport=3000
