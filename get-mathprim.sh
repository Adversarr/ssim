MATHPRIM_GITLINK=https://github.com/Adversarr/mathprim.git
WARNING_COLOR='\033[0;31m'
INFO_COLOR='\033[0;34m'
NO_COLOR='\033[0m'

# ensure git, cmake
if ! command -v git &>/dev/null; then
  echo "git could not be found"
  exit
fi

if ! command -v cmake &>/dev/null; then
  echo "cmake could not be found"
  exit
fi

# If exists, remove mathprim and _build
if [ -d "mathprim" ]; then
  cd mathprim
  echo -e "${INFO_COLOR}Pulling the latest version of mathprim to $(pwd)...${NO_COLOR}"
  git pull &>/dev/null
  if [ $? -ne 0 ]; then
    echo -e "${WARNING_COLOR}Failed to pull the latest version of mathprim.${NO_COLOR}"
    exit
  fi
  # show version
  git log -1 --pretty=format:"%h %s"
  echo -e "${INFO_COLOR}Git pull finished.${NO_COLOR}"
  cd ..
else
  echo -e "${INFO_COLOR}Cloning mathprim...${NO_COLOR}"
  git clone $MATHPRIM_GITLINK mathprim
fi
if [ -d "_build" ]; then
  rm -rf _build
fi
if [ -d "installed" ]; then
  rm -rf installed
fi

CMAKE_ENABLE_CUDA=OFF
# Test nvcc.
if [ -n "$(command -v nvcc)" ]; then
  CMAKE_ENABLE_CUDA=ON
  echo -e "${INFO_COLOR}nvcc found. CUDA will be enabled.${NO_COLOR}"
else
  echo -e "${INFO_COLOR}nvcc not found. CUDA will be disabled.${NO_COLOR}"
fi

echo -e "${INFO_COLOR}Configuring mathprim...${NO_COLOR}"
cmake -S mathprim -B _build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_INSTALL_PREFIX=$(pwd)/installed \
  -DMATHPRIM_BUILD_TESTS=OFF \
  -DMATHPRIM_ENABLE_OPENMP=ON \
  -DMATHPRIM_ENABLE_CUDA=$CMAKE_ENABLE_CUDA \
  -DMATHPRIM_ENABLE_WARNINGS=OFF \
  -DMATHPRIM_BUILD_TESTS=OFF \
  -DMATHPRIM_INSTALL=ON > /dev/null

if [ $? -ne 0 ]; then
  echo -e "${WARNING_COLOR}Failed to configure mathprim.${NO_COLOR}"
  exit
fi

echo -e "${INFO_COLOR}Build&Install mathprim...${NO_COLOR}"
cmake --build _build >/dev/null
if [ $? -ne 0 ]; then
  echo -e "${WARNING_COLOR}Failed to build mathprim.${NO_COLOR}"
  exit
fi
cmake --install _build >/dev/null
if [ $? -ne 0 ]; then
  echo -e "${WARNING_COLOR}Failed to install mathprim.${NO_COLOR}"
  exit
fi
echo -e "${INFO_COLOR}mathprim installed to $(pwd)/installed.${NO_COLOR}"
echo "remove _build directories? ([y]/n)"
read -r answer

if [ "$answer" = "y" ] || [ "$answer" = "" ]; then
  rm -rf _build
  echo -e "${INFO_COLOR}Remove _build.${NO_COLOR}"
fi

echo -e "${INFO_COLOR}Done.${NO_COLOR}"
