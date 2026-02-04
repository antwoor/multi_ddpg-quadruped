#!/bin/sh
set -e
set -u

PROJECTNAME="antwoor/rl:stable"
HOMENAME=$(basename "$HOME")

display_flag=""
gpus=""

usage() {
  echo "Usage: $0 -g <gpus> -d <true/false>" 1>&2
  exit 1
}

while getopts "g:d:" opt; do
  case $opt in
    g) gpus="$OPTARG" ;;
    d) display_flag="$OPTARG" ;;
    ?) usage ;;
  esac
done

shift $((OPTIND - 1))

errors=""

if [ -z "$gpus" ]; then
  errors="$errors""GPUs parameter is required.\n"
elif ! echo "$gpus" | grep -Eq '^[1-9]+$|^all$'; then
  errors="$errors""Invalid GPUs parameter. Must be a numeric value or 'all'.\n"
fi

if [ -z "$display_flag" ]; then
  errors="$errors""Display parameter is required.\n"
fi

if [ -n "$errors" ]; then
  printf "%s" "$errors" >&2
  usage
else
  printf "Use Docker Image: %s\n" "$PROJECTNAME"
  printf "Start Docker Container: %s_container\n" "$PROJECTNAME"
  printf "Use GPU: %s\n" "$gpus"
  printf "Use Display: %s\n\n" "$display_flag"
fi

if [ "$display_flag" = "true" ]; then
  export DISPLAY="$DISPLAY"
  printf "Setting display to %s\n" "$DISPLAY"
  xhost +
  docker run -it --rm \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "/home/${HOMENAME}/.Xauthority:/root/.Xauthority" \
    -e "DISPLAY=$DISPLAY" \
    -v "${PWD}:/home/root/rl_ws" \
    --network=host \
    --gpus="$gpus" \
    --name="ppo_container" \
    "$PROJECTNAME" /bin/bash
else
  printf "Running Docker without display\n"
  docker run -it --rm \
    -v "${PWD}/src:/home/root/rl_ws" \
    --network=host \
    --gpus="$gpus" \
    --name="ppo_container" \
    "$PROJECTNAME" /bin/bash
fi