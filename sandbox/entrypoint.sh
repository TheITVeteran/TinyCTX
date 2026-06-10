#!/bin/sh
# sandbox/entrypoint.sh
# Runs as root. Applies firewall rules then starts the server.
# The server runs as root inside this locked-down container:
#   - cap_drop: ALL (NET_ADMIN only during this script, dropped after exec)
#   - read_only filesystem
#   - no host network access
#   - network-isolated from agent and LAN
set -e

# 1. Always allow loopback and established/related (replies to inbound connections).
iptables -A OUTPUT -o lo                                        -j ACCEPT
iptables -A OUTPUT -m conntrack --ctstate ESTABLISHED,RELATED   -j ACCEPT
# 2. Block private, reserved, and link-local egress.
iptables -A OUTPUT -d 0.0.0.0/8          -j DROP  # "this" network
iptables -A OUTPUT -d 10.0.0.0/8         -j DROP  # RFC 1918
iptables -A OUTPUT -d 100.64.0.0/10      -j DROP  # CGNAT (Tailscale etc.)
iptables -A OUTPUT -d 169.254.0.0/16     -j DROP  # link-local / cloud metadata
iptables -A OUTPUT -d 172.16.0.0/12      -j DROP  # RFC 1918
iptables -A OUTPUT -d 192.0.0.0/24       -j DROP  # IETF protocol assignments
iptables -A OUTPUT -d 192.0.2.0/24       -j DROP  # TEST-NET-1
iptables -A OUTPUT -d 192.168.0.0/16     -j DROP  # RFC 1918
iptables -A OUTPUT -d 198.18.0.0/15      -j DROP  # benchmarking
iptables -A OUTPUT -d 198.51.100.0/24    -j DROP  # TEST-NET-2
iptables -A OUTPUT -d 203.0.113.0/24     -j DROP  # TEST-NET-3
iptables -A OUTPUT -d 240.0.0.0/4        -j DROP  # reserved
iptables -A OUTPUT -d 255.255.255.255/32 -j DROP  # broadcast
# IPv6 — block private/reserved ranges via ip6tables (installed with the iptables package on Debian).
ip6tables -A OUTPUT -o lo                                        -j ACCEPT
ip6tables -A OUTPUT -m conntrack --ctstate ESTABLISHED,RELATED   -j ACCEPT
ip6tables -A OUTPUT -d ::1/128           -j DROP  # loopback
ip6tables -A OUTPUT -d ::ffff:0:0/96     -j DROP  # IPv4-mapped
ip6tables -A OUTPUT -d 64:ff9b::/96      -j DROP  # NAT64 well-known prefix
ip6tables -A OUTPUT -d 100::/64          -j DROP  # discard-only
ip6tables -A OUTPUT -d 2002::/16         -j DROP  # 6to4 (embeds IPv4)
ip6tables -A OUTPUT -d fc00::/7          -j DROP  # ULA
ip6tables -A OUTPUT -d fe80::/10         -j DROP  # link-local
ip6tables -A OUTPUT -d ff00::/8          -j DROP  # multicast
echo "[firewall] egress rules installed"

exec python -m sandbox
