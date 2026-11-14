#!/bin/sh
# Mantiq for Linux and macOS - ./setup.sh, another name for Mantiq's installer (./mntq.sh): the first run sets
# everything up and starts the dashboard; later runs just start it.  ./setup.sh --help lists the options.
cd "$(dirname "$0")" || exit 1
exec ./mntq.sh "$@"
