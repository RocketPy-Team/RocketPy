######################
RocketPy Documentation
######################

**Version**: |release|

**Useful links**:
`Installation <https://docs.rocketpy.org/en/latest/user/installation.html>`_ |
`Source Repository <https://github.com/RocketPy-Team/RocketPy>`_ |
`Issue Tracker <https://github.com/RocketPy-Team/RocketPy/issues>`_ |
`Q&A Support <https://discord.gg/b6xYnNh>`_ |

RocketPy is the next-generation trajectory simulation solution for High-Power Rocketry. The code is written as a Python library and allows for a complete 6 degrees of freedom simulation of a rocket's flight trajectory, including high-fidelity variable mass effects as well as descent under parachutes. Weather conditions, such as wind profiles, can be imported from sophisticated datasets, allowing for realistic scenarios. Furthermore, the implementation facilitates complex simulations, such as multi-stage rockets, design and trajectory optimization and dispersion analysis.

.. grid:: 2

    .. grid-item-card::
       :img-top: ./static/landing_images/getting_started.svg

       Getting started
       ^^^^^^^^^^^^^^^

       Simulating your first rocket? Check out the Beginner's Guide. It contains an
       introduction to RocketPy main concepts and walks you through the process of
       setting up a simulation.

       +++

       .. button-ref:: user/first_simulation
          :expand:
          :color: secondary
          :click-parent:

          To the beginner's guide

    .. grid-item-card::
       :img-top: ./static/landing_images/user_guide.svg
       
       User guide
       ^^^^^^^^^^

       The user guide provides in-depth information on
       RocketPy functionalities with useful background information and explanation.

       +++

       .. button-ref:: user/index
          :expand:
          :color: secondary
          :click-parent:

          To the user guide

    .. grid-item-card::
       :img-top: ./static/landing_images/api.svg

       API reference
       ^^^^^^^^^^^^^

       The reference guide contains a detailed description of RocketPy modules. Here it is described how the methods work and which parameters can be used. It assumes that you already have an understanding of key concepts.

       +++

       .. button-ref:: reference/index
          :expand:
          :color: secondary
          :click-parent:

          To the reference guide

    .. grid-item-card::
       :img-top: ./static/landing_images/contributor.svg

       Contributor's guide
       ^^^^^^^^^^^^^^^^^^^

       Want to contribute to RocketPy source code? The contributing guidelines will guide you through the process of improving RocketPy.

       +++

       .. button-ref:: development/index
          :expand:
          :color: secondary
          :click-parent:

          To the contributor's guide

.. toctree::
   :maxdepth: 1
   :hidden:

   User Guide <user/index>
   Code Reference <reference/index>
   Development <development/index>
   Technical <technical/index>
   Flight Examples <examples/index>

