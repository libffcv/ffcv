conda create -n ffcv python=3.9
conda activate ffcv

echo "Activated!"

conda install compilers pkg-config libjpeg-turbo -c conda-forge
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch-nightly
# conda install pytorch torchvision==0.10 torchaudio cudatoolkit=11.3 compilers pkg-config libjpeg-turbo opencv -c pytorch-nightly -c conda-forge
