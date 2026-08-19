# Create `iterative-separation.sif` file
NAME="iterative-separation"
singularity build --fakeroot ~/$NAME.sif $NAME.def
mv ~/$NAME.sif ./