#!/bin/bash

source /environment.sh

# initialize launch file
dt_launchfile_init

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------

CATKIN_WS="/code/catkin_ws"
APRILTAG_PACKAGE_DIR="${CATKIN_WS}/src/dt-apriltag-detector"
DATA_DIR="${APRILTAG_PACKAGE_DIR}/dataset_generator"
TESTS_DIR="${APRILTAG_PACKAGE_DIR}/dataset_generator/tests"
CAMPARAMFILE="datasheet.csv"
VIEWER_TEMPLATE="double"
VIEWER_PATH="${APRILTAG_PACKAGE_DIR}/packages/viewer_templates/${VIEWER_TEMPLATE}.html"

if [ -n "$EXPERIMENT" ]; then
  EXPERIMENTS=("_${EXPERIMENT}")
else
  EXPERIMENTS=("raw" "rect")
fi

CAMPARAMS="${DATA_DIR}/$CAMPARAMFILE"
[ ! -f $CAMPARAMS ] && { echo "$CAMPARAMS file not found"; exit 99; }

for EXP in "${EXPERIMENTS[@]}"; do

  # constants
  DATA_TYPE="${EXP}"

  # test configuration
  IN_FOLDER="${DATA_DIR}/${DATA_TYPE}/*"
  i=0
  for INPUT in $IN_FOLDER; do

    while IFS=, read -r idx tag_id tvec rvec fx fy cx cy k1 k2 p1 p2 k3 p11 p12 p13 p21 p22 p23 p31 p32 p33; do
      [[ "$fx" != "fx" ]] && FX="$fx"
      [[ "$fy" != "fy" ]] && FY="$fy"
      [[ "$cx" != "cx" ]] && CX="$cx"
      [[ "$cy" != "cy" ]] && CY="$cy"

      [[ "$k1" != "k1" ]] && K1="$k1"
      [[ "$k2" != "k2" ]] && K2="$k2"
      [[ "$p1" != "p1" ]] && P1="$p1"
      [[ "$p2" != "p2" ]] && P2="$p2"
      [[ "$k3" != "k3" ]] && K3="$k3"

      [[ "$p11" != "p11" ]] && P11="$p11"
      [[ "$p12" != "p12" ]] && P12="$p12"
      [[ "$p13" != "p13" ]] && P13="$p13"
      [[ "$p21" != "p21" ]] && P21="$p21"
      [[ "$p22" != "p22" ]] && P22="$p22"
      [[ "$p23" != "p23" ]] && P23="$p23"
      [[ "$p31" != "p31" ]] && P31="$p31"
      [[ "$p32" != "p32" ]] && P32="$p32"
      [[ "$p33" != "p33" ]] && P33="$p33"

    done < $CAMPARAMS

    TEST_IMAGE="${INPUT}"
    IN_FILE="${INPUT}"
    SAMPLENR="sample_$i"
    i=$((i+1))

    OUT_DIR="${TESTS_DIR}/${SAMPLENR}/${EXP}"
    # make out dir
    mkdir -p "${OUT_DIR}"
    if [ -z "$FX" ] 
    then
      continue
    else
      # launching app
      (
        cd ${OUT_DIR} || exit;
        ${CATKIN_WS}/devel/bin/apriltag_demo \
            --debug \
            --decimate=1.0 \
            --refine-edges=0 \
            --fx=$FX \
            --fy=$FY \
            --cx=$CX \
            --cy=$CY \
            --k1=$K1 \
            --k2=$K2 \
            --p1=$P1 \
            --p2=$P2 \
            --k3=$K3 \
            --p11=$P11 \
            --p12=$P12 \
            --p13=$P13 \
            --p21=$P21 \
            --p22=$P22 \
            --p23=$P23 \
            --p31=$P31 \
            --p32=$P32 \
            --p33=$P33 \
            ${IN_FILE};
            mogrify -format jpg ./*.pnm;
            rm -f ./*.pnm
      )
    fi
    i=i+1
  done

  # create viewer
  VIEWER_OUT="${OUT_DIR}/../viewer.html"
  cp ${VIEWER_PATH} "${VIEWER_OUT}"
  sed -i "s/__EXP1__/raw/g" "${VIEWER_OUT}"
  sed -i "s/__EXP2__/rect/g" "${VIEWER_OUT}"
  sed -i "s/__EXPERIMENT__/${DATA_TYPE}/g" "${VIEWER_OUT}"

done

# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE

# terminate launch file
dt_launchfile_terminate
