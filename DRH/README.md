<!-- TABLE OF CONTENTS -->
## Overview
This folder documents the the analysis of DRH data reported in "Inferring Cultural Landscapes with the Inverse Ising Model" (mainly Section 5). Code-base currently being cleaned and made reproducible. 

* Preprocessing of the DRH data. 
* Data curation for the subset used in the paper. 
* Table creation (tables X-X). 
* Figures 3 and 4. 


<!-- ABOUT THE PROJECT -->
## Files

* ```preprocessing.py``` converts ```.json``` obtained from the DRH to ```.csv```.
* ```curation.py``` runs data curation before ```MPF``` (see also: ```run_curation.sh```).
* ```plot_parameters.py``` creates figure 3A and figure 3B. 
* ```plot_configurations.py``` creates figure 4A.
* ```seed_methodist.py``` creates figure 4B. 
* ```seed_roman.py``` creates figure 4C.


<!-- GETTING STARTED -->
## Getting Started

Environments tested on ubuntu version 22.04 LTS. 

### Requirements 

Working installation of ```Python``` (tested with v3.10.6) and ```Julia``` (tested with vXXX).

### Installation


1. Clone the repo (if not already done). Here shown for ```ssh``` but ```https``` also fine:
    ```sh
    git clone git@github.com:victor-m-p/humanities-glass.git
    ```

2. Install the ```Python``` environment (```glassenv```):
    ```sh
    bash create_venv.sh
    bash add_venv.sh
    ```

3. Install the ```Julia``` environment  


<!-- USAGE EXAMPLES -->
## 

<!-- LICENSE -->
## License
NB: MOVE TO OVERALL README. 
Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Victor Poulsen - [@vic_moellervpoulsen](https://twitter.com/vic_moeller) - victormoeller@gmail.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

conIII
 


* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)
* [React Icons](https://react-icons.github.io/react-icons/search)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

