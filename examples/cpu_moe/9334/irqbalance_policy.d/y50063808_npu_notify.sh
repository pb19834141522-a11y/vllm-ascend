#!/bin/sh
# Only this NPU's message notification vector is owned by this policy.
case "$1" in
  */0000:11:00.0)
    if awk -v irq="$2:" '$1 == irq && $0 ~ /IR-PCI-MSIX-0000:11:00[.]0[[:space:]]+2-edge[[:space:]]+non-trans msg_chan_rx_msg_notify/ { found=1 } END { exit !found }' /proc/interrupts; then
      printf 'ban=true\n'
      exit 0
    fi
    ;;
esac
exit 1
