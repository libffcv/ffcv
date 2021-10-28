conda create -n ffcv python=3.9
conda activate ffcv
conda install pytorch torchvision torchaudio cudatoolkit=11.3 compilers pkg-config opencv libjpeg-turbo -c pytorch -c conda-forge
conda install numba
