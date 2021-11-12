import gdown
import os

gdown.download('https://drive.google.com/uc?id=1_8ZCtB4DAaIBg856NBUVT9IhXwO72RVA')

if os.name != 'nt':
    os.system('unzip emnist.zip')
    os.system('rm -rf emnist.zip')
