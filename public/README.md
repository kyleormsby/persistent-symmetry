# Intro/Table of Contents

The goal of this repository is to explore how we can use persistent homology to detect rotational and translation symmetry in images. The code relies on Ripser to run persistent homology computations.

The [public](https://github.com/kyleormsby/persistent-symmetry/tree/main/public) folder is organized as follows: 
1. [1_wallpaper_homology.ipynb](https://github.com/kyleormsby/persistent-symmetry/blob/main/public/1_wallpaper_homology.ipynb) contains necessary background on the topology of wallpaper manifolds
2. [2_phom_E(2)_mod_gamma.ipynb](https://github.com/kyleormsby/persistent-symmetry/blob/main/public/2_phom_E(2)_mod_gamma.ipynb) defines distance metrics on parameterizations on these wallpaper manifolds and applies persistent homology to test if the persistent homology symmetry detection method is theoretically sound 
3. [3_translations_only_PH.ipynb](https://github.com/kyleormsby/persistent-symmetry/blob/main/public/3_translations_only_PH.ipynb) demonstrates that persistent homology can successfully detect translational symmetries in images
4. [4_phom_identification.ipynb](https://github.com/kyleormsby/persistent-symmetry/blob/main/public/4_phom_identification.ipynb) demonstrates how the persistent homology symmetry detection algorithm breaks down in some cases when rotational symmetry is added
5. [5_alternative_embeddings_distances.ipynb](https://github.com/kyleormsby/persistent-symmetry/blob/main/public/5_alternative_embeddings_distances.ipynb) covers several methods that we used to try to fix where the algorithm breaks down 
6. [6_compare_idealized_distance_matrix.ipynb](https://github.com/kyleormsby/persistent-symmetry/blob/main/public/6_compare_idealized_distance_matrix.ipynb) explores how the distance metrics on images differs from the distances we defined in [2_phom_E(2)_mod_gamma.ipynb](https://github.com/kyleormsby/persistent-symmetry/blob/main/public/2_phom_E(2)_mod_gamma.ipynb) 
A1. [A1_ translation_lattices.ipynb](https://github.com/kyleormsby/persistent-symmetry/blob/main/public/A1_%20translation_lattices.ipynb) demonstrates that other, less-involved algorithms that do not rely on persistent homology can be used to detect translational symmetry in images
A2. [A2_determining_point_group.ipynb](https://github.com/kyleormsby/persistent-symmetry/blob/main/public/A2_determining_point_group.ipynb) similarly demonstrates that other, less-involved algorithms that do not rely on persistent homology can be used to detect rotational symmetry in images.

