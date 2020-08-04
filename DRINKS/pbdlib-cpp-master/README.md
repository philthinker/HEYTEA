# pbdlib-cpp

<p>PbDlib is a collection of C++ source codes for robot programming by demonstration (learning from demonstration). It includes various functionalities at the crossroad of statistical learning, dynamical systems, optimal control and Riemannian geometry.</p>
<p>PbDlib can be used in applications requiring task adaptation, human-robot skill transfer, safe controllers based on minimal intervention principle, as well as for probabilistic motion analysis and synthesis in multiple coordinate systems.</p>
<p>Other versions of the library in Matlab (compatible with GNU Octave) and Python are also available at http://www.idiap.ch/software/pbdlib/ (currently, the Matlab version has the most functionalities).</p>


### Usage

This project builds a number of executables in the `build` folder, with the corresponding C++ source codes in the `src` folder listed in the table below. The corresponding publications are also listed below. 


### List of examples

| Filename | ref. | Description |
|----------|------|-------------|
| [demo_ergodicControl_2D01.cpp](./src/demo_ergodicControl_2D01.cpp) | [[5]](#ref-5) | 2D ergodic control with spectral multiscale coverage (SMC) algorithm |
| [demo_ergodicControl_nD01.cpp](./src/demo_ergodicControl_nD01.cpp) | [[5]](#ref-5) | nD ergodic control with spectral multiscale coverage (SMC) algorithm |
| [demo_GMR01.cpp](./src/demo_GMR01.cpp) | [[5]](#ref-5) | Gaussian mixture model (GMM) and time-based Gaussian mixture regression (GMR) used for reproduction |
| [demo_GPR01.cpp](./src/demo_GPR01.cpp) | [[2]](#ref-2) | Gaussian process regression (GPR) |
| [demo_HSMM_batchLQR01.cpp](./src/demo_HSMM_batchLQR01.cpp) | [[2]](#ref-2) | Use of HSMM (with lognormal duration model) and batch LQR (with position only) for motion synthesis |
| [demo_infHorLQR01.cpp](./src/demo_infHorLQR01.cpp) | [[2]](#ref-2) | Discrete infinite horizon linear quadratic regulation |
| [demo_LWR_batch01.cpp](./src/demo_LWR_batch01.cpp) | [[2]](#ref-2) | Locally weighted regression (LWR) with radial basis functions (RBF), using batch computation |
| [demo_LWR_iterative01.cpp](./src/demo_LWR_iterative01.cpp) | [[2]](#ref-2) | Locally weighted regression (LWR) with radial basis functions (RBF), using iterative computation |
| [demo_MPC_batch01.cpp](./src/demo_MPC_batch01.cpp) | [[2]](#ref-2), [[7]](#ref-7) | Model predictive control (MPC) with batch linear quadratic tracking (LQT) formulation |
| [demo_MPC_iterative01.cpp](./src/demo_MPC_iterative01.cpp) | [[2]](#ref-2), [[7]](#ref-7) | Model predictive control (MPC) with iterative linear quadratic tracking (LQT) formulation |
| [demo_MPC_semitied01.cpp](./src/demo_MPC_semitied01.cpp) | [[2]](#ref-2), [[7]](#ref-7) | MPC with semi-tied covariances |
| [demo_MPC_velocity01.cpp](./src/demo_MPC_velocity01.cpp) | [[2]](#ref-2), [[7]](#ref-7) | MPC with an objective including velocity tracking |
| [demo_online_GMM01.cpp](./src/demo_online_GMM01.cpp) | [[2]](#ref-2), [[6]](#ref-6) | Online GMM learning and LQR-based trajectory generation |
| [demo_online_HSMM01.cpp](./src/demo_online_HSMM01.cpp) | [[2]](#ref-2), [[6]](#ref-6) | Online HSMM learning and sampling with LQR-based trajectory generation |
| [demo_proMP01.cpp](./src/demo_proMP01.cpp) | [[5]](#ref-5) | Conditioning on trajectory distributions with ProMP |
| [demo_Riemannian_pose_batchLQR01.cpp](./src/demo_Riemannian_pose_batchLQR01.cpp) | [[3]](#ref-3) | Linear quadratic regulation of pose by relying on Riemannian manifold and batch LQR (MPC without constraints) |
| [demo_Riemannian_pose_GMM01.cpp](./src/demo_Riemannian_pose_GMM01.cpp) | [[3]](#ref-3) | GMM for 3D position and unit quaternion data by relying on Riemannian manifold |
| [demo_Riemannian_pose_infHorLQR01.cpp](./src/demo_Riemannian_pose_infHorLQR01.cpp) | [[3]](#ref-3) | Linear quadratic regulation of pose by relying on Riemannian manifold and infinite-horizon LQR |
| [demo_Riemannian_pose_TPGMM01.cpp](./src/demo_Riemannian_pose_TPGMM01.cpp) | [[3]](#ref-3) | TP-GMM of pose (R3 x S3) with two frames |
| [demo_Riemannian_S2_GMM01.cpp](./src/demo_Riemannian_S2_GMM01.cpp) | [[3]](#ref-3) | GMM for data on a sphere by relying on Riemannian manifold |
| [demo_Riemannian_S2_infHorLQR01.cpp](./src/demo_Riemannian_S2_infHorLQR01.cpp) | [[3]](#ref-3) | Linear quadratic regulation on a sphere by relying on Riemannian manifold and infinite-horizon LQR |
| [demo_Riemannian_S2_product01.cpp](./src/demo_Riemannian_S2_product01.cpp) | [[3]](#ref-3) | Gaussian product on sphere |
| [demo_Riemannian_S2_TPGMM01.cpp](./src/demo_Riemannian_S2_TPGMM01.cpp) | [[3]](#ref-3) | TP-GMM for data on a sphere by relying on Riemannian manifold (with single frame) |
| [demo_Riemannian_S3_infHorLQR01.cpp](./src/demo_Riemannian_S3_infHorLQR01.cpp) | [[3]](#ref-3) | Linear quadratic regulation on hypersphere (orientation as unit quaternions) by relying on Riemannian manifold and infinite-horizon LQR |
| [demo_Riemannian_S3_TPGMM01.cpp](./src/demo_Riemannian_S3_TPGMM01.cpp) | [[3]](#ref-3) | TP-GMM on S3 (unit quaternion) with two frames |
| [demo_Riemannian_SPD_GMR01.cpp](./src/demo_Riemannian_SPD_GMR01.cpp) | [[4]](#ref-4) | GMR with time as input and covariance data as output by relying on Riemannian manifold |
| [demo_Riemannian_SPD_interp02.cpp](./src/demo_Riemannian_SPD_interp02.cpp) | [[4]](#ref-4) | Covariance interpolation on Riemannian manifold from a GMM with augmented covariances |
| [demo_TPGMMProduct01.cpp](./src/demo_TPGMMProduct01.cpp) | [[1]](#ref-1) | Product of Gaussians for a Task-Parametrized GMM |
| [demo_TPGMR01.cpp](./src/demo_TPGMR01.cpp) | [[1]](#ref-1) | Task-Parametrized GMM with GMR (time as input), the model is able to adapt to continuously changing task parameters. |
| [demo_TPMPC01.cpp](./src/demo_TPMPC01.cpp) | [[1]](#ref-1) | Linear quadratic control (unconstrained linear MPC) acting in multiple frames, which is equivalent to the fusion of Gaussian controllers |


### References

If you find PbDlib useful for your research, please cite the related publications!

<p><a name="ref-1">
[1] Calinon, S. (2016). <strong>A Tutorial on Task-Parameterized Movement Learning and Retrieval</strong>. Intelligent Service Robotics (Springer), 9:1, 1-29.
</a><br>
[[pdf]](http://calinon.ch/papers/Calinon-JIST2015.pdf)
[[bib]](http://calinon.ch/papers/Calinon-JIST2015.bib)
<br><strong>(Ref. for GMM, TP-GMM, MFA, MPPCA, GPR, trajGMM)</strong>
</p>

<p><a name="ref-2">
[2] Calinon, S. and Lee, D. (2019). <strong>Learning Control</strong>. Vadakkepat, P. and Goswami, A. (eds.). Humanoid Robotics: a Reference, pp. 1261-1312. Springer.
</a><br>
[[pdf]](http://calinon.ch/papers/Calinon-Lee-learningControl.pdf)
[[bib]](http://calinon.ch/papers/Calinon-Lee-learningControl.bib)
<br><strong>(Ref. for MPC, LQR, HMM, HSMM)</strong>
</p>

<p><a name="ref-3">
[3] Calinon, S. and Jaquier, N. (2019). <strong>Gaussians on Riemannian Manifolds for Robot Learning and Adaptive Control</strong>. arXiv:1909.05946.
</a><br>
[[pdf]](http://calinon.ch/papers/Calinon-arXiv2019.pdf)
[[bib]](http://calinon.ch/papers/Calinon-arXiv2019.bib)
<br><strong>(Ref. for Riemannian manifolds)</strong>
</p>

<p><a name="ref-4">
[4] Jaquier, N. and Calinon, S. (2017). <strong>Gaussian Mixture Regression on Symmetric Positive Definite Matrices Manifolds: Application to Wrist Motion Estimation with sEMG</strong>. In Proc. of the IEEE/RSJ Intl Conf. on Intelligent Robots and Systems (IROS), pp. 59-64.
</a><br>
[[pdf]](http://calinon.ch/papers/Jaquier-IROS2017.pdf)
[[bib]](http://calinon.ch/papers/Jaquier-IROS2017.bib)
<br><strong>(Ref. for S^+ Riemannian manifolds)</strong>
</p>

<p><a name="ref-5">
[5] Calinon, S. (2019). <strong>Mixture Models for the Analysis, Edition, and Synthesis of Continuous Time Series</strong>. Bouguila, N. and Fan, W. (eds). Mixture Models and Applications, pp. 39-57. Springer.
</a><br>
[[pdf]](http://calinon.ch/papers/Calinon_MMchapter2019.pdf)
[[bib]](http://calinon.ch/papers/Calinon_MMchapter2019.bib)
<br><strong>(Ref. for ergodic control, Bezier curves, LWR, GMR, probabilistic movement primitives)</strong>
</p>

<p><a name="ref-6">
[6] Bruno, D., Calinon, S. and Caldwell, D.G. (2017). <strong>Learning Autonomous Behaviours for the Body of a Flexible Surgical Robot</strong>. Autonomous Robots, 41:2, 333-347.
</a><br>
[[pdf]](http://calinon.ch/papers/Bruno-AURO2017.pdf)
[[bib]](http://calinon.ch/papers/Bruno-AURO2017.bib)
<br><strong>(Ref. for DP-means)</strong>
</p>

<p><a name="ref-7">
[7] Berio, D., Calinon, S. and Fol Leymarie, F. (2017). <strong>Generating Calligraphic Trajectories with Model Predictive Control</strong>. In Proc. of the 43rd Conf. on Graphics Interface, pp. 132-139.
</a><br>
[[pdf]](http://calinon.ch/papers/Berio-GI2017.pdf)
[[bib]](http://calinon.ch/papers/Berio-GI2017.bib)
<br><strong>(Ref. for Bezier curves as MPC with viapoints)</strong>
</p>

<p><a name="ref-8">
[8] EPFL EE613 course "Machine Learning for Engineers"
</a><br>
[[url]](http://calinon.ch/teaching.htm)
<br><strong>(Ref. for machine learning teaching material)</strong>
</p>


### Installation prerequisite

PbDlib requires:

  - *glfw3* (lightweight and portable library for managing OpenGL contexts, windows and inputs)
  - *GLEW* (The OpenGL Extension Wrangler Library)
  - *LAPACK* (Linear Algebra PACKage)
  - *Armadillo* (C++ library for linear algebra & scientific computing)

Instructions are given below.

*ImGui* (graphical user interface library for C++, [https://github.com/ocornut/imgui](https://github.com/ocornut/imgui)) and
*gfx_ui* (a minimal geometry editing UI, [https://github.com/colormotor/gfx\_ui](https://github.com/colormotor/gfx_ui)), are also
used and provided as part of this package. They are both distributed under the *MIT license* (see the docs/ folder).


### Dependencies installation on Debian and Ubuntu

#### glfw3

```
sudo apt-get install libglfw3-dev
```

**Installation from source**

If libglfw3-dev is not available on your system, you can install it manually with:

```
git clone https://github.com/glfw/glfw.git
cd glfw
mkdir build
cd build
cmake -DBUILD_SHARED_LIBS=ON ../
make
sudo make install
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
```

*Note:* The `-DBUILD_SHARED_LIBS` is necessary otherwise cmake will create a
static library.

#### GLEW

```
sudo apt-get install libglew-dev
```

#### LAPACK

```
sudo apt-get install liblapack-dev
```

#### Armadillo

See [http://arma.sourceforge.net/download.html](http://arma.sourceforge.net/download.html)
for instructions.


### Compilation (on Linux and MacOS)

```
cd pbdlib-cpp
mkdir build
cd build
cmake ..
make
```


### Compilation (on Windows)

First, download all the necessary dependencies. They are (but Armadillo) provided as
a **binary package** on their own homepage. We recommend to download and extract them
all in the same folder.

**glfw3**: https://www.glfw.org/download.html <br>
**GLEW**: http://glew.sourceforge.net/index.html <br>
**Armadillo**: http://arma.sourceforge.net/download.html <br>

Then use cmake-gui (http://cmake.org) to generate a solution for you version of Visual
Studio. You'll need to manually indicate the path to the following dependencies when
requested:

  * `ARMADILLO_DIR`: Path to Armadillo
  * `GLEW_DIR`: Path to GLEW
  * `GLFW_DIR`: Path to GLFW

Once the solution has been generated, open it (`pbdlib_gui.sln` in your build folder)
and compile as usual.


### Gallery

|                         |                         |
|-------------------------|-------------------------|
| ![](https://gitlab.idiap.ch/rli/pbdlib-cpp-sandbox/raw/master/images/demo_ergodicControl_2D01.gif) <br> [demo\_demo\_ergodicControl\_2D01.cpp](./src/demo_ergodicControl_2D01.cpp) | ![](https://gitlab.idiap.ch/rli/pbdlib-cpp-sandbox/raw/master/images/demo_GMR01.gif) <br> [demo\_GMR01.cpp](./src/demo_GMR01.cpp) |
| ![](https://gitlab.idiap.ch/rli/pbdlib-cpp-sandbox/raw/master/images/demo_GPR01.gif) <br> [demo\_GPR01.cpp](./src/demo_GPR01.cpp) | ![](https://gitlab.idiap.ch/rli/pbdlib-cpp-sandbox/raw/master/images/demo_HSMM_batchLQR01.gif) <br> [demo\_HSMM\_batchLQR01.cpp](./src/demo_HSMM_batchLQR01.cpp) |
| ![](https://gitlab.idiap.ch/rli/pbdlib-cpp-sandbox/raw/master/images/demo_LWR_batch01.gif) <br> [demo\_LWR\_batch01.cpp](./src/demo_LWR_batch01.cpp) | ![](https://gitlab.idiap.ch/rli/pbdlib-cpp-sandbox/raw/master/images/demo_LWR_iterative01.gif) <br> [demo\_LWR\_iterative01.cpp](./src/demo_LWR_iterative01.cpp) |
| ![](https://gitlab.idiap.ch/rli/pbdlib-cpp-sandbox/raw/master/images/demo_MPC_batch01.gif) <br> [demo\_MPC\_batch01.cpp](./src/demo_MPC_batch01.cpp) | ![](https://gitlab.idiap.ch/rli/pbdlib-cpp-sandbox/raw/master/images/demo_MPC_iterative01.gif) <br> [demo\_MPC\_iterative01.cpp](./src/demo_MPC_iterative01.cpp) |
| ![](https://gitlab.idiap.ch/rli/pbdlib-cpp-sandbox/raw/master/images/demo_MPC_semitied01.gif) <br> [demo\_MPC\_semitied01.cpp](./src/demo_MPC_semitied01.cpp) | ![](https://gitlab.idiap.ch/rli/pbdlib-cpp-sandbox/raw/master/images/demo_MPC_velocity01.gif) <br> [demo\_MPC\_velocity01.cpp](./src/demo_MPC_velocity01.cpp) |
| ![](https://gitlab.idiap.ch/rli/pbdlib-cpp-sandbox/raw/master/images/demo_online_GMM01.gif) <br> [demo\_online\_GMM01.cpp](./src/demo_online_GMM01.cpp) | ![](https://gitlab.idiap.ch/rli/pbdlib-cpp-sandbox/raw/master/images/demo_proMP01.gif) <br> [demo\_proMP01.cpp](./src/demo_proMP01.cpp) |
| ![](https://gitlab.idiap.ch/rli/pbdlib-cpp-sandbox/raw/master/images/demo_Riemannian_SPD_GMR01.gif) <br> [demo\_Riemannian\_SPD\_GMR01.cpp](./src/demo_Riemannian_SPD_GMR01.cpp) | ![](https://gitlab.idiap.ch/rli/pbdlib-cpp-sandbox/raw/master/images/demo_Riemannian_SPD_interp02.gif) <br> [demo\_Riemannian\_SPD\_interp02.cpp](./src/demo_Riemannian_SPD_interp02.cpp) | 
| ![](https://gitlab.idiap.ch/rli/pbdlib-cpp-sandbox/raw/master/images/demo_Riemannian_pose_batchLQR01.gif) <br> [demo\_Riemannian\_pose\_batchLQR01.cpp](./src/demo_Riemannian_pose_batchLQR01.cpp) | ![](https://gitlab.idiap.ch/rli/pbdlib-cpp-sandbox/raw/master/images/demo_Riemannian_pose_infHorLQR01.gif) <br> [demo\_Riemannian\_pose\_infHorLQR01.cpp](./src/demo_Riemannian_pose_infHorLQR01.cpp) |
| ![](https://gitlab.idiap.ch/rli/pbdlib-cpp-sandbox/raw/master/images/demo_Riemannian_S3_infHorLQR01.png) <br> [demo\_Riemannian\_S3\_infHorLQR01.cpp](./src/demo_Riemannian_S3_infHorLQR01.cpp) | ![](https://gitlab.idiap.ch/rli/pbdlib-cpp-sandbox/raw/master/images/demo_Riemannian_S3_TPGMM01.gif) <br> [demo\_Riemannian\_S3\_TPGMM01.cpp](./src/demo_Riemannian_S3_TPGMM01.cpp) |
| ![](https://gitlab.idiap.ch/rli/pbdlib-cpp-sandbox/raw/master/images/demo_Riemannian_S2_GMM01.gif) <br> [demo\_Riemannian\_S2\_GMM01.cpp](./src/demo_Riemannian_S2_GMM01.cpp) | ![](https://gitlab.idiap.ch/rli/pbdlib-cpp-sandbox/raw/master/images/demo_Riemannian_S2_infHorLQR01.gif) <br> [demo\_Riemannian\_S2\_infHorLQR01.cpp](./src/demo_Riemannian_S2_infHorLQR01.cpp) | 
| ![](https://gitlab.idiap.ch/rli/pbdlib-cpp-sandbox/raw/master/images/demo_Riemannian_S2_product01.gif) <br> [demo\_Riemannian\_S2\_product01.cpp](./src/demo_Riemannian_S2_product01.cpp) | ![](https://gitlab.idiap.ch/rli/pbdlib-cpp-sandbox/raw/master/images/demo_Riemannian_S2_TPGMM01.gif) <br> [demo\_Riemannian\_S2\_TPGMM01.cpp](./src/demo_Riemannian_S2_TPGMM01.cpp) |
| ![](https://gitlab.idiap.ch/rli/pbdlib-cpp-sandbox/raw/master/images/demo_TPMPC01.gif) <br> [demo\_TPMPC01.cpp](./src/demo_TPMPC01.cpp) | ![](https://gitlab.idiap.ch/rli/pbdlib-cpp-sandbox/raw/master/images/demo_TPGMMProduct01.gif) <br> [demo\_TPGMMProduct01.cpp](./src/demo_TPGMMProduct01.cpp) | 
| ![](https://gitlab.idiap.ch/rli/pbdlib-cpp-sandbox/raw/master/images/demo_TPGMR01.gif) <br> [demo\_TPGMR01.cpp](./src/demo_TPGMR01.cpp) | |

