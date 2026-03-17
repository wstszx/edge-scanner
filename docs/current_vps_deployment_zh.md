# Edge Scanner 当前 VPS 部署 Runbook

这份文档整理的是当前已经验证可用的一套线上配置，目标是以后重装、迁移或排障时可以直接照着恢复。

适用范围：

- Ubuntu VPS
- 仓库部署目录固定为 `/root/edge-scanner`
- 继续使用 `root` 用户部署
- Web 和后台扫描拆成两个 systemd 服务
- Nginx 对外提供 HTTP/HTTPS
- 公网 `443` 需要和既有的 x-ui/xray 类 TLS 服务共存
- GitHub Actions 通过 SSH 私钥自动发布
- 当前不使用 `VPS_HOST_FINGERPRINT`

## 当前线上拓扑

- 代码目录：`/root/edge-scanner`
- Python 虚拟环境：`/root/edge-scanner/venv`
- Web 服务：`edge-scanner.service`
- 后台扫描服务：`edge-scanner-worker.service`
- 反向代理：`nginx`
- Gunicorn 监听：`127.0.0.1:5000`
- Edge Scanner 本地 TLS 入口：`127.0.0.1:8443`
- 既有代理服务本地 TLS 入口：`127.0.0.1:1443`
- 对外访问域名：`edge.wstszx.de5.net`

当前设计要点：

- Flask 开发服务器不用于生产流量
- Web 和自动扫描拆开跑，避免多 worker 复制后台线程
- 公网不再直接暴露 `5000`
- Nginx 在 `443` 上按 SNI 分流，避免直接挤掉原有代理服务

## Cloudflare DNS

当前有效配置：

- 新增 `A` 记录：`edge -> 198.46.158.36`
- `edge` 不配置 `AAAA`，除非未来真的启用该 VPS 的 IPv6

建议：

- 源站 HTTPS 正常后，Cloudflare 可以切橙云
- Cloudflare `SSL/TLS` 选择 `Full (strict)`
- 不要用 `Flexible`

## 首次部署或重建 VPS

安装基础依赖：

```bash
apt update
apt install -y python3 python3-pip python3-venv git nginx certbot python3-certbot-nginx
```

拉取代码并创建虚拟环境：

```bash
cd /root
git clone https://github.com/wstszx/edge-scanner.git
cd /root/edge-scanner
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

应用配置文件：

- 项目环境变量文件位置：`/root/edge-scanner/.env`
- 如果有 Odds API key，就写在 `.env`
- 自动扫描建议只在 worker 服务里启用

## Systemd 服务

直接使用仓库里的 root 模板：

```bash
cp /root/edge-scanner/deploy/systemd/root/edge-scanner.service /etc/systemd/system/edge-scanner.service
cp /root/edge-scanner/deploy/systemd/root/edge-scanner-worker.service /etc/systemd/system/edge-scanner-worker.service

systemctl daemon-reload
systemctl enable edge-scanner edge-scanner-worker
systemctl restart edge-scanner edge-scanner-worker
```

校验：

```bash
systemctl status edge-scanner --no-pager
systemctl status edge-scanner-worker --no-pager
```

说明：

- `edge-scanner.service` 只跑 Web，不承担自动扫描
- `edge-scanner-worker.service` 负责后台自动扫描
- 两个服务共享同一份代码和 `.env`

## Nginx 与共享 443

如果公网 `443` 没有被其他服务占用，可用常规模板：

- `deploy/nginx/edge-scanner.conf`

当前线上实际使用的是共享 `443` 方案，因为原有代理服务已占用 TLS 能力。使用这两个模板：

- `deploy/nginx/edge-scanner-shared-443-site.conf`
- `deploy/nginx/edge-scanner-shared-443-stream.conf`

核心思路：

- Nginx 对外监听 `80` 和 `443`
- `edge.wstszx.de5.net` 的 TLS 流量转到 `127.0.0.1:8443`
- 其他 TLS 流量默认转到既有代理服务的 `127.0.0.1:1443`

落地步骤：

```bash
mkdir -p /var/www/certbot /etc/nginx/stream-conf.d

cp /root/edge-scanner/deploy/nginx/edge-scanner-shared-443-site.conf /etc/nginx/sites-available/edge-scanner.conf
cp /root/edge-scanner/deploy/nginx/edge-scanner-shared-443-stream.conf /etc/nginx/stream-conf.d/edge-scanner.conf

rm -f /etc/nginx/sites-enabled/default
ln -sf /etc/nginx/sites-available/edge-scanner.conf /etc/nginx/sites-enabled/edge-scanner.conf
nginx -t
systemctl restart nginx
```

如果 `/etc/nginx/nginx.conf` 还没有 `stream` 块，需要补上：

```nginx
stream {
    include /etc/nginx/stream-conf.d/*.conf;
}
```

注意：

- 既有 x-ui/xray 入站不要再绑定公网 `443`
- 保持它只监听本地端口，例如 `127.0.0.1:1443`
- Edge Scanner 的本地 TLS 端口保持为 `127.0.0.1:8443`

## HTTPS 证书

共享 `443` 方案下，证书申请优先使用 webroot：

```bash
certbot certonly --webroot -w /var/www/certbot -d edge.wstszx.de5.net
```

添加续签后的 Nginx reload 钩子：

```bash
mkdir -p /etc/letsencrypt/renewal-hooks/deploy
printf '%s\n' '#!/bin/sh' 'systemctl reload nginx' > /etc/letsencrypt/renewal-hooks/deploy/reload-nginx.sh
chmod +x /etc/letsencrypt/renewal-hooks/deploy/reload-nginx.sh
```

校验证书：

```bash
echo | openssl s_client -servername edge.wstszx.de5.net -connect 127.0.0.1:443 2>/dev/null | openssl x509 -noout -subject -issuer -dates
```

## GitHub Actions 自动发布

当前仓库使用：

- workflow 文件：`.github/workflows/deploy.yml`
- 发布目标分支：`main`

当前有效配置：

必填 GitHub Secrets：

- `VPS_HOST=198.46.158.36`
- `VPS_SSH_KEY=<SSH 私钥完整内容>`

可选 GitHub Secrets：

- `VPS_SSH_USER=root`

当前不使用：

- `VPS_HOST_FINGERPRINT`

说明：

- `VPS_SSH_KEY` 必须是私钥，不是 root 密码，也不是 `.pub` 公钥
- 当前 workflow 允许不配置 `VPS_HOST_FINGERPRINT`
- 如果以后重新启用指纹校验，必须确认 GitHub Actions 实际协商到的 host key 指纹与 Secret 完全一致

当前 root 布局下，一般不需要额外配置 GitHub Variables，因为 workflow 已带默认值：

- `DEPLOY_REPO_DIR=/root/edge-scanner`
- `DEPLOY_VENV_PYTHON=/root/edge-scanner/venv/bin/python`
- `DEPLOY_WEB_SERVICE=edge-scanner`
- `DEPLOY_WORKER_SERVICE=edge-scanner-worker`

自动发布流程实际执行的是：

```bash
cd /root/edge-scanner
git fetch --prune origin
git checkout <github.sha>
/root/edge-scanner/venv/bin/python -m pip install -r requirements.txt
systemctl daemon-reload
systemctl restart edge-scanner
systemctl restart edge-scanner-worker
```

## 日常更新

代码变更发布：

1. 本地提交并推送到 `main`
2. GitHub Actions 自动 SSH 到 VPS
3. VPS checkout 到这次 push 的精确 commit
4. 自动安装依赖并重启服务

文档变更说明：

- 当前 workflow 对 `**/*.md` 和 `docs/**` 配了 `paths-ignore`
- 纯文档提交不会自动触发线上发布
- 如果只改文档，不需要重启 VPS

如果需要手动更新 VPS：

```bash
cd /root/edge-scanner
git fetch origin
git checkout main
git pull --ff-only
source venv/bin/activate
pip install -r requirements.txt
systemctl restart edge-scanner edge-scanner-worker
```

## 验证命令

服务状态：

```bash
systemctl status edge-scanner --no-pager
systemctl status edge-scanner-worker --no-pager
systemctl status nginx --no-pager
```

端口监听：

```bash
ss -ltnp | egrep ':(80|443|5000|8443|1443) '
```

本机回环验证：

```bash
curl -I http://127.0.0.1:5000/
curl -I http://127.0.0.1/
curl -I https://edge.wstszx.de5.net/
```

日志查看：

```bash
journalctl -u edge-scanner -f
journalctl -u edge-scanner-worker -f
journalctl -u nginx -f
```

## 当前有效结论

- 生产 Web 入口使用 Gunicorn，不再使用 Flask dev server
- 后台自动扫描独立为 `edge-scanner-worker.service`
- `edge.wstszx.de5.net` 是当前对外访问入口
- GitHub Actions 通过 SSH 私钥自动发布
- 当前不启用 `VPS_HOST_FINGERPRINT`
- 纯文档提交不会触发自动部署
