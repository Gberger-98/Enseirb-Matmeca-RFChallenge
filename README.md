# Single-Channel RF Challenge Starter Code

[Click here for details on the challenge setup](https://rfchallenge.mit.edu/wp-content/uploads/2021/08/Challenge1_pdf_detailed_description.pdf)

The helper functions in this starter code use SigMF and CommPy. To install these dependencies:
```bash
pip install git+https://github.com/gnuradio/SigMF.git
pip install scikit-commpy==0.6.0
```
(The current code and dataset depends on scikit-commpy version 0.6.0; using the latest version may introduce inconsistencies with the files generated in the dataset. Future versions of the starter code and dataset will account for the updates in this dependency.)

Other dependencies include: NumPy, Matplotlib, Tensorflow (to run the [`bitregression`](https://github.com/RFChallenge/rfchallenge_singlechannel_starter/tree/main/example/demod_bitregression) example), tqdm (for progress bar)  

Please ensure that the `dataset` is saved in the folder "dataset", and retains the folder hierarchy as provided -- this allows the helper functions to correctly find and load the corresponding files.

To obtain the dataset, you may use the following commands (from within the RFChallenge starter code folder):
```bash
wget -O rfc_dataset.zip "https://www.dropbox.com/s/clh4xq7u3my3nx6/rfc_dataset.zip?dl=0"
jar -xvf rfc_dataset.zip
rm rfc_dataset.zip
```
(Note that `jar` is used here, instead of `unzip`, since the zip file is larger than 4 GB. You may also use `7z` to extract the contents.)


The python notebook [`notebook/PlotBer.ipynb`](https://github.com/Gberger-98/Enseirb-Matmeca-RFChallenge/blob/main/notebook/PlotBer.ipynb) illustrate our result.

---
### Direct Download Links:
* [Dataset (Training set)](https://www.dropbox.com/s/clh4xq7u3my3nx6/rfc_dataset.zip?dl=0) (Latest Version: Jun 15, 2021)
* [CNN 1D (model)](https://www.dropbox.com/s/54gezgxlysc2irx/models.rar?dl=0) (Latest Version: Jan 26, 2022)
