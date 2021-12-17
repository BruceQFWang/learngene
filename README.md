# Learngene: From Open-World to Your Learning Task

If you use the code in this repo for your work, please cite the following bib entries:

    @misc{wang2021learngene,
      title={Learngene: From Open-World to Your Learning Task}, 
      author={Qiufeng Wang and Xin Geng and Shuxia Lin and Shiyu Xia and Lei Qi and Ning Xu},
      year={2021},
      eprint={2106.06788},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
    }
 
 ## Abstract
Although deep learning has made significant progress on fixed large-scale datasets, it typically encounters challenges regarding improperly detecting unknown/unseen classes in the open-world scenario, over-parametrized, and overfitting small samples. Since biological systems can overcome the above difficulties very well, individuals inherit an innate gene from collective creatures that have evolved over hundreds of millions of years and then learn new skills through few examples. Inspired by this, we propose a practical collective-individual paradigm where an evolution (expandable) network is trained on sequential tasks and then recognize unknown classes in real-world. Moreover, the learngene, i.e., the gene for learning initialization rules of the target model, is proposed to inherit the meta-knowledge from the collective model and reconstruct a lightweight individual model on the target task. Particularly, a novel criterion is proposed to discover learngene in the collective model, according to the gradient information. Finally, the individual model is trained only with few samples on the target learning tasks. We demonstrate the effectiveness of our approach in an extensive empirical study and theoretical analysis.

## Make dataset
Data division refers to [appendix](https://github.com/BruceQFWang/learngene/blob/main/Learngene_Appendix.pdf)

Make continual data (source domain) and target data(target domain) on the CIFAR100 dataset:

    $ cd utils
    $ python data_cifar_mk.py --num_imgs_per_cat_train 600 --path [name of data path]
    
Make continual data (source domain) and target data(target domain) on the ImageNet-100 dataset:

    $ cd utils
    $ python data_imagenet_mk.py --path [name of data path]
    
## Generate collective model
Train collective-model on the CIFAR100 dataset:
    $ cd collective-model
    $ python val_lifelong_cifar100.py --batch-size 64 --epochs 50 --num_works 50 --path [name of continualdataset path]
