#!/bin/bash
#SBATCH --job-name=make_index_files
#SBATCH --output=../logs/make_index_files_%a.out
#SBATCH --error=../logs/make_index_files_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G
#SBATCH --array=0-7,9

source ../env/bin/activate
cd ..

if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then

    # NOTE - must already have source tar file at $DATASRC/ILSVRC2012_img_train.tar
    source="ilsvrc_2012"
    mkdir -p ${SLURM_TMPDIR}/${source}
    cp -r ./data/meta_dataset/records/${source} $SLURM_TMPDIR

    find ${SLURM_TMPDIR} -name '*.tfrecords' -type f \
                        -exec sh -c 'python -m src.datasets.meta_dataset.tfrecord.tools.tfrecord2idx $2 ${2%.tfrecords}.index' sh ${SLURM_TMPDIR} {} \;

    cd $SLURM_TMPDIR
    tar -zcvf ${source}.tar.gz ${source}
    cp -r ${source}.tar.gz $HOME/scratch-new/
    echo "Successfully Converted ilsvrc_2012"
fi

if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then

    source="omniglot"
    mkdir -p ${SLURM_TMPDIR}/${source}
    cp -r /network/projects/r/ramnath.kumar/meta_dataset/records/${source} $SLURM_TMPDIR

    find ${SLURM_TMPDIR} -name '*.tfrecords' -type f \
                        -exec sh -c 'python -m src.datasets.meta_dataset.tfrecord.tools.tfrecord2idx $2 ${2%.tfrecords}.index' sh ${SLURM_TMPDIR} {} \;

    cd $SLURM_TMPDIR
    tar -zcvf ${source}.tar.gz ${source}
    cp -r ${source}.tar.gz $HOME/scratch-new/
    echo "Successfully Converted omniglot"
fi

if [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then

    source="aircraft"
    mkdir -p ${SLURM_TMPDIR}/${source}
    cp -r /network/projects/r/ramnath.kumar/meta_dataset/records/${source} $SLURM_TMPDIR

    find ${SLURM_TMPDIR} -name '*.tfrecords' -type f \
                        -exec sh -c 'python -m src.datasets.meta_dataset.tfrecord.tools.tfrecord2idx $2 ${2%.tfrecords}.index' sh ${SLURM_TMPDIR} {} \;

    cd $SLURM_TMPDIR
    tar -zcvf ${source}.tar.gz ${source}
    cp -r ${source}.tar.gz $HOME/scratch-new/
    echo "Successfully Converted aircraft"
fi

if [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then

    source="cu_birds"
    mkdir -p ${SLURM_TMPDIR}/${source}
    cp -r /network/projects/r/ramnath.kumar/meta_dataset/records/${source} $SLURM_TMPDIR

    find ${SLURM_TMPDIR} -name '*.tfrecords' -type f \
                        -exec sh -c 'python -m src.datasets.meta_dataset.tfrecord.tools.tfrecord2idx $2 ${2%.tfrecords}.index' sh ${SLURM_TMPDIR} {} \;

    cd $SLURM_TMPDIR
    tar -zcvf ${source}.tar.gz ${source}
    cp -r ${source}.tar.gz $HOME/scratch-new/
    echo "Successfully Converted aircraft"
fi

if [ $SLURM_ARRAY_TASK_ID -eq 4 ]; then

    source="dtd"
    mkdir -p ${SLURM_TMPDIR}/${source}
    cp -r /network/projects/r/ramnath.kumar/meta_dataset/records/${source} $SLURM_TMPDIR

    find ${SLURM_TMPDIR} -name '*.tfrecords' -type f \
                        -exec sh -c 'python -m src.datasets.meta_dataset.tfrecord.tools.tfrecord2idx $2 ${2%.tfrecords}.index' sh ${SLURM_TMPDIR} {} \;

    cd $SLURM_TMPDIR
    tar -zcvf ${source}.tar.gz ${source}
    cp -r ${source}.tar.gz $HOME/scratch-new/
    echo "Successfully Converted dtd"
fi

if [ $SLURM_ARRAY_TASK_ID -eq 5 ]; then

    source="quickdraw"
    mkdir -p ${SLURM_TMPDIR}/${source}
    cp -r /network/projects/r/ramnath.kumar/meta_dataset/records/${source} $SLURM_TMPDIR

    find ${SLURM_TMPDIR} -name '*.tfrecords' -type f \
                        -exec sh -c 'python -m src.datasets.meta_dataset.tfrecord.tools.tfrecord2idx $2 ${2%.tfrecords}.index' sh ${SLURM_TMPDIR} {} \;

    cd $SLURM_TMPDIR
    tar -zcvf ${source}.tar.gz ${source}
    cp -r ${source}.tar.gz $HOME/scratch-new/
    echo "Successfully Converted quickdraw"
fi

if [ $SLURM_ARRAY_TASK_ID -eq 6 ]; then

    source="fungi"
    mkdir -p ${SLURM_TMPDIR}/${source}
    cp -r /network/projects/r/ramnath.kumar/meta_dataset/records/${source} $SLURM_TMPDIR

    find ${SLURM_TMPDIR} -name '*.tfrecords' -type f \
                        -exec sh -c 'python -m src.datasets.meta_dataset.tfrecord.tools.tfrecord2idx $2 ${2%.tfrecords}.index' sh ${SLURM_TMPDIR} {} \;

    cd $SLURM_TMPDIR
    tar -zcvf ${source}.tar.gz ${source}
    cp -r ${source}.tar.gz $HOME/scratch-new/
    echo "Successfully Converted fungi"
fi

if [ $SLURM_ARRAY_TASK_ID -eq 7 ]; then

    source="vgg_flower"
    mkdir -p ${SLURM_TMPDIR}/${source}
    cp -r /network/projects/r/ramnath.kumar/meta_dataset/records/${source} $SLURM_TMPDIR

    find ${SLURM_TMPDIR} -name '*.tfrecords' -type f \
                        -exec sh -c 'python -m src.datasets.meta_dataset.tfrecord.tools.tfrecord2idx $2 ${2%.tfrecords}.index' sh ${SLURM_TMPDIR} {} \;

    cd $SLURM_TMPDIR
    tar -zcvf ${source}.tar.gz ${source}
    cp -r ${source}.tar.gz $HOME/scratch-new/
    echo "Successfully Converted vgg_flower"
fi

if [ $SLURM_ARRAY_TASK_ID -eq 8 ]; then

    source="traffic_sign"
    mkdir -p ${SLURM_TMPDIR}/${source}
    cp -r /network/projects/r/ramnath.kumar/meta_dataset/records/${source} $SLURM_TMPDIR

    find ${SLURM_TMPDIR} -name '*.tfrecords' -type f \
                        -exec sh -c 'python -m src.datasets.meta_dataset.tfrecord.tools.tfrecord2idx $2 ${2%.tfrecords}.index' sh ${SLURM_TMPDIR} {} \;

    cd $SLURM_TMPDIR
    tar -zcvf ${source}.tar.gz ${source}
    cp -r ${source}.tar.gz $HOME/scratch-new/
    echo "Successfully Converted traffic_sign"
fi

if [ $SLURM_ARRAY_TASK_ID -eq 9 ]; then

    source="mscoco"
    mkdir -p ${SLURM_TMPDIR}/${source}
    cp -r /network/projects/r/ramnath.kumar/meta_dataset/records/${source} $SLURM_TMPDIR

    find ${SLURM_TMPDIR} -name '*.tfrecords' -type f \
                        -exec sh -c 'python -m src.datasets.meta_dataset.tfrecord.tools.tfrecord2idx $2 ${2%.tfrecords}.index' sh ${SLURM_TMPDIR} {} \;

    cd $SLURM_TMPDIR
    tar -zcvf ${source}.tar.gz ${source}
    cp -r ${source}.tar.gz $HOME/scratch-new/
    echo "Successfully Converted mscoco"
fi
