conda create -n ffcv python=3.9
conda activate ffcv

echo "Activated!"

conda install pytorch torchvision torchaudio cudatoolkit=11.3 compilers pkg-config opencv libjpeg-turbo -c pytorch -c conda-forge
