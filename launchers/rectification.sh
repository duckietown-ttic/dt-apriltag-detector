#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------

CATKIN_WS="/code/catkin_ws"
APRILTAG_PACKAGE_DIR="${CATKIN_WS}/src/dt-apriltag-detector"
DATA_DIR="${APRILTAG_PACKAGE_DIR}/packages/data"
TESTS_DIR="${APRILTAG_PACKAGE_DIR}/packages/tests"
VIEWER_TEMPLATE="double"
VIEWER_PATH="${APRILTAG_PACKAGE_DIR}/packages/viewer_templates/${VIEWER_TEMPLATE}.html"

if [ -n "$EXPERIMENT" ]; then
  EXPERIMENTS=("_${EXPERIMENT}")
else
  EXPERIMENTS=("_center" "_left" "_right")
fi


for EXP in "${EXPERIMENTS[@]}"; do

  # constants
  DATA_TYPE="rectification${EXP}"

  for INPUT in "raw" "rect"; do

    # test configuration
    TEST_IMAGE="${INPUT}.jpg"
    TEST_NAME="rectification${EXP}/${INPUT}"
    IN_FILE="${DATA_DIR}/${DATA_TYPE}/${TEST_IMAGE}"
    OUT_DIR="${TESTS_DIR}/${TEST_NAME}"

    # make out dir
    mkdir -p "${OUT_DIR}"

    # launching app
    (
      cd ${OUT_DIR} || exit;
      ${CATKIN_WS}/devel/bin/apriltag_demo \
          --debug \
          --decimate=1.0 \
          --refine-edges=0 \
          ${IN_FILE};
      mogrify -format jpg ./*.pnm;
      rm -f ./*.pnm
    )

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

# wait for app to end
dt-launchfile-join