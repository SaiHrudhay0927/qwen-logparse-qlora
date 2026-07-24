"""Deterministic generator for the log->JSON dataset (seed 42).
Produces data/{train,validation,test}.jsonl — identical to the notebook's split.
"""
import json, random, ipaddress, os

SEED = 42
N_TRAIN, N_VAL, N_TEST = 2000, 200, 300
rng = random.Random(SEED)

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
USERS  = ["admin","root","ubuntu","svc_backup","jenkins","postgres","test","oracle",
          "s.parna","dev01","guest","ftpuser","nagios","www-data"]
HOSTS  = ["web-01","db-prod-2","bastion","edge-gw","app-07","mail-01"]
CMDS   = ["/usr/bin/apt update","/bin/cat /etc/shadow","/usr/sbin/service nginx restart",
          "/bin/rm -rf /var/log/auth.log","/usr/bin/systemctl stop firewalld","/bin/chmod 777 /opt"]
PATHS  = ["/index.php","/admin/login","/api/v1/users","/wp-admin/","/.env","/static/app.js",
          "/health","/api/v2/transactions"]
METHODS = ["GET","POST","PUT","DELETE","HEAD"]
STATUS  = [200,201,301,302,400,401,403,404,500,502]

def rand_ip():      return str(ipaddress.IPv4Address(rng.randint(1<<24,(1<<32)-1)))
def rand_priv_ip(): return f"10.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}"
def ts_syslog():    return f"{rng.choice(MONTHS)} {rng.randint(1,28):2d} {rng.randint(0,23):02d}:{rng.randint(0,59):02d}:{rng.randint(0,59):02d}"
def ts_clf():       return f"{rng.randint(1,28):02d}/{rng.choice(MONTHS)}/2026:{rng.randint(0,23):02d}:{rng.randint(0,59):02d}:{rng.randint(0,59):02d} +0530"

def gen_ssh_fail():
    u,ip,port,host=rng.choice(USERS),rand_ip(),rng.randint(1024,65535),rng.choice(HOSTS)
    inv=rng.random()<0.5
    return (f"{ts_syslog()} {host} sshd[{rng.randint(1000,9999)}]: Failed password for "
            f"{'invalid user ' if inv else ''}{u} from {ip} port {port} ssh2",
            {"event":"failed_login","service":"ssh","user":u,"src_ip":ip,"port":port})
def gen_ssh_ok():
    u,ip,port,host=rng.choice(USERS),rand_ip(),rng.randint(1024,65535),rng.choice(HOSTS)
    m=rng.choice(["password","publickey"])
    return (f"{ts_syslog()} {host} sshd[{rng.randint(1000,9999)}]: Accepted {m} for {u} from {ip} port {port} ssh2",
            {"event":"successful_login","service":"ssh","user":u,"src_ip":ip,"port":port})
def gen_sudo():
    u,cmd,host=rng.choice(USERS),rng.choice(CMDS),rng.choice(HOSTS)
    return (f"{ts_syslog()} {host} sudo: {u} : TTY=pts/{rng.randint(0,4)} ; PWD=/home/{u} ; USER=root ; COMMAND={cmd}",
            {"event":"privilege_escalation","service":"sudo","user":u,"command":cmd})
def gen_nginx():
    ip,m,p,s=rand_ip(),rng.choice(METHODS),rng.choice(PATHS),rng.choice(STATUS)
    return (f'{ip} - - [{ts_clf()}] "{m} {p} HTTP/1.1" {s} {rng.randint(120,90000)} "-" "Mozilla/5.0"',
            {"event":"http_request","service":"nginx","src_ip":ip,"method":m,"path":p,"status":s})
def gen_ufw():
    src,dst,dport,host=rand_ip(),rand_priv_ip(),rng.choice([22,23,80,443,3306,3389,8080,5432]),rng.choice(HOSTS)
    return (f"{ts_syslog()} {host} kernel: [UFW BLOCK] IN=eth0 OUT= MAC=00:16:3e:aa:bb:cc "
            f"SRC={src} DST={dst} LEN={rng.randint(40,1500)} PROTO=TCP SPT={rng.randint(1024,65535)} DPT={dport}",
            {"event":"firewall_block","service":"ufw","src_ip":src,"dst_ip":dst,"dst_port":dport})
def gen_windows():
    u,ip=rng.choice(USERS),rand_ip();lt=rng.choice([2,3,10])
    return (f"EventID=4625 An account failed to log on. Subject: Security ID: NULL SID  "
            f"Account Name: {u}  Logon Type: {lt}  Source Network Address: {ip}  "
            f"Failure Reason: Unknown user name or bad password.",
            {"event":"failed_login","service":"windows","user":u,"src_ip":ip,"logon_type":lt})

GENERATORS=[gen_ssh_fail,gen_ssh_ok,gen_sudo,gen_nginx,gen_ufw,gen_windows]

def make(n):
    seen,out=set(),[]
    while len(out)<n:
        line,gold=rng.choice(GENERATORS)()
        if line in seen: continue
        seen.add(line); out.append({"log":line,"json":json.dumps(gold)})
    return out

if __name__=="__main__":
    data=make(N_TRAIN+N_VAL+N_TEST)
    os.makedirs("data",exist_ok=True)
    for name,rows in {"train":data[:N_TRAIN],
                      "validation":data[N_TRAIN:N_TRAIN+N_VAL],
                      "test":data[N_TRAIN+N_VAL:]}.items():
        with open(f"data/{name}.jsonl","w") as f:
            for r in rows: f.write(json.dumps(r)+"\n")
    print(f"wrote {len(data)} examples ({len(set(r['log'] for r in data))} unique)")
