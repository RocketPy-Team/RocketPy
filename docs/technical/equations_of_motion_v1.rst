======================
Equations of Motion v1
======================

Introduction
------------

This document briefly describes the equations of motion which govern the flight
of a rocket used in v1.0 onwards. This document simply shows some of the 
algebraic steps used to get to the final form of the equations of motion used
in the code. For a more detailed explanation of the equations of motion, please
refer to :ref:`Equations of Motion v0 <eqsv0>`.

Development
-----------

**Linear Equation:**

.. math:: 

   m \left( \dot{\mathbf{v}} + \dot{\omega} \times \mathbf{r}_{\mathrm{CM}} \right. & \left.+\boldsymbol{\omega} \times\left(\boldsymbol{\omega} \times \mathbf{r}_{\mathrm{CM}} \right) + \mathbf{r}_{CM}^{\prime \prime}+2 \boldsymbol{\omega} \times \mathbf{r}_{\mathrm{CM}}^{\prime}\right) \\
   = & \mathbf{T} - 2 \dot{m} \mathbf{r}_{\mathrm{CM}}^{\prime}+2 \boldsymbol{\omega} \times \dot{m}\left(\mathbf{r}_{\mathrm{noz}}-\mathbf{r}_{\mathrm{CM}}\right)+\ddot{m}\left(\mathbf{r}_{\mathrm{noz}}-\mathbf{r}_{\mathrm{CM}}\right)+\mathbf{A}+\sum_i \mathbf{N}_i-m g \hat{\mathbf{a}}_3


**Angular Equation:**

.. math::
   
   \mathbf{I} \cdot \dot{\boldsymbol{\omega}}+\boldsymbol{\omega} \times(\mathbf{I} \cdot \boldsymbol{\omega})+\mathbf{I}^{\prime} \cdot \boldsymbol{\omega}+m \mathbf{r}_{\mathrm{CM}} \times \dot{\mathbf{v}}=\left(\dot{m} \mathbf{S}_{\mathrm{noz}}\right) \cdot \boldsymbol{\omega}+\sum_i \mathbf{r}_i \times \mathbf{N}_i-\mathbf{r}_{\mathrm{CM}} \times m g \hat{\mathbf{a}}_3

**Cross multiplying the linear equation by** :math:`r_{CM}` **:**

.. math::

   \begin{aligned}
   m \mathbf{r}_{\mathrm{CM}} \times \dot{\mathbf{v}}+m \mathbf{r}_{\mathrm{CM}} \times\left(\dot{\boldsymbol{\omega}} \times \mathbf{r}_{\mathrm{CM}}\right)+m \mathbf{r}_{\mathrm{CM}} \times \boldsymbol{\omega} \times\left(\boldsymbol{\omega} \times \mathbf{r}_{\mathrm{CM}}\right)+m \mathbf{r}_{\mathrm{CM}} \times \mathbf{r}_{\mathrm{CM}}^{\prime \prime}+2 m \mathbf{r}_{\mathrm{CM}} \times \boldsymbol{\omega} \times \mathbf{r}_{\mathrm{CM}}^{\prime} \\
   \quad=\mathbf{r}_{\mathrm{CM}} \times \mathbf{T}-2 \dot{m} \mathbf{r}_{\mathrm{CM}} \times \mathbf{r}_{\mathrm{CM}}^{\prime}+2 \mathbf{r}_{\mathrm{CM}} \times \boldsymbol{\omega} \times \dot{m}\left(\mathbf{r}_{\mathrm{noz}}-\mathbf{r}_{\mathrm{CM}}\right)+m \mathbf{r}_{\mathrm{CM}}^{\prime} \times\left(\mathbf{r}_{\mathrm{noz}}-\mathbf{r}_{\mathrm{CM}}\right) \\
   \quad+\mathbf{r}_{\mathrm{CM}} \times \mathbf{A}+\mathbf{r}_{\mathrm{CM}} \times \sum_i \mathbf{N}_i-m \mathbf{r}_{\mathrm{CM}} \times g \hat{\mathbf{a}}_3
   \end{aligned}

**Simplifying:**

.. math::
   
   \mathbf{r}_{\mathrm{CM}} \times\left(\dot{\omega} \times \mathbf{r}_{\mathrm{CM}}\right)=\left(\mathbf{r}_{\mathrm{CM}} \cdot \mathbf{r}_{\mathrm{CM}}\right) \dot{\boldsymbol{\omega}}-\left(\mathbf{r}_{\mathrm{CM}} \cdot \dot{\boldsymbol{\omega}}\right) \mathbf{r}_{\mathrm{CM}}

   \mathbf{r}_{\mathrm{CM}} \times \boldsymbol{\omega} \times\left(\boldsymbol{\omega} \times \mathbf{r}_{\mathrm{CM}}\right)=\mathbf{r}_{\mathrm{CM}} \times\left(\left(\boldsymbol{\omega} \cdot \mathbf{r}_{\mathrm{CM}}\right) \boldsymbol{\omega}-(\boldsymbol{\omega} \cdot \boldsymbol{\omega}) \mathbf{r}_{\mathrm{CM}}\right)=\left(\boldsymbol{\omega} \cdot \mathbf{r}_{\mathrm{CM}}\right) \mathbf{r}_{\mathrm{CM}} \times \boldsymbol{\omega}

.. math::

   \begin{aligned}
   m \mathbf{r}_{\mathrm{CM}} \times \dot{\mathbf{v}}=- & m\left(\mathbf{r}_{\mathrm{CM}} \cdot \mathbf{r}_{\mathrm{CM}}\right) \dot{\boldsymbol{\omega}}+m\left(\mathbf{r}_{\mathrm{CM}} \cdot \dot{\boldsymbol{\omega}}\right) \mathbf{r}_{\mathrm{CM}}-m\left(\boldsymbol{\omega} \cdot \mathbf{r}_{\mathrm{CM}}\right) \mathbf{r}_{\mathrm{CM}} \times \boldsymbol{\omega}-m \mathbf{r}_{\mathrm{CM}} \times \mathbf{r}_{\mathrm{CM}}^{\prime \prime} \\
   & -2 m \mathbf{r}_{\mathrm{CM}} \times \boldsymbol{\omega} \times \mathbf{r}_{\mathrm{CM}}^{\prime}+\mathbf{r}_{\mathrm{CM}} \times \mathbf{T}-2 \dot{m} \mathbf{r}_{\mathrm{CM}} \times \mathbf{r}_{\mathrm{CM}}^{\prime}+2 \mathbf{r}_{\mathrm{CM}} \times \boldsymbol{\omega} \times \dot{m}\left(\mathbf{r}_{\mathrm{noz}}-\mathbf{r}_{\mathrm{CM}}\right) \\
   & +\ddot{m} \mathbf{r}_{\mathrm{CM}} \times\left(\mathbf{r}_{\mathrm{noz}}-\mathbf{r}_{\mathrm{CM}}\right)+\mathbf{r}_{\mathrm{CM}} \times \mathbf{A}+\mathbf{r}_{\mathrm{CM}} \times \sum_i \mathbf{N}_i-m \mathbf{r}_{\mathrm{CM}} \times g \hat{\mathbf{a}}_3
   \end{aligned}

**Substituting in the Angular equation:**

.. math::

   \begin{aligned}
   \mathbf{I} \cdot \dot{\boldsymbol{\omega}}+\boldsymbol{\omega} \times(\mathbf{I} \cdot \boldsymbol{\omega})+\mathbf{I}^{\prime} \cdot \boldsymbol{\omega}-m\left(\mathbf{r}_{\mathrm{CM}} \cdot \mathbf{r}_{\mathrm{CM}}\right) \dot{\boldsymbol{\omega}}+m\left(\mathbf{r}_{\mathrm{CM}} \cdot \dot{\boldsymbol{\omega}}\right) \mathbf{r}_{\mathrm{CM}}-m\left(\boldsymbol{\omega} \cdot \mathbf{r}_{\mathrm{CM}}\right) \mathbf{r}_{\mathrm{CM}} \times \boldsymbol{\omega}-m \mathbf{r}_{\mathrm{CM}} \times \mathbf{r}_{\mathrm{CM}}^{\prime \prime} \\
   -2 m \mathbf{r}_{\mathrm{CM}} \times \boldsymbol{\omega} \times \mathbf{r}_{\mathrm{CM}}^{\prime}+\mathbf{r}_{\mathrm{CM}} \times \mathbf{T}-2 \dot{m} \mathbf{r}_{\mathrm{CM}} \times \mathbf{r}_{\mathrm{CM}}^{\prime}+2 \mathbf{r}_{\mathrm{CM}} \times \boldsymbol{\omega} \times \dot{m}\left(\mathbf{r}_{\mathrm{noz}}-\mathbf{r}_{\mathrm{CM}}\right) \\
   +\ddot{m} \mathbf{r}_{\mathrm{CM}} \times\left(\mathbf{r}_{\mathrm{noz}}-\mathbf{r}_{\mathrm{CM}}\right)+\mathbf{r}_{\mathrm{CM}} \times \mathbf{A}+\mathbf{r}_{\mathrm{CM}} \times \sum_i \mathbf{N}_i-m \mathbf{r}_{\mathrm{CM}} \times g \hat{\mathbf{a}}_3 \\
   =\left(\dot{m} \mathbf{S}_{\mathrm{noz}}\right) \cdot \boldsymbol{\omega}+\sum_i \mathbf{r}_i \times \mathbf{N}_i-\mathbf{r}_{\mathrm{CM}} \times m g \hat{\mathbf{a}}_3
   \end{aligned}

.. math::

   \begin{aligned}
   & \mathbf{I} \cdot \dot{\boldsymbol{\omega}}-m\left(\mathbf{r}_{\mathrm{CM}} \cdot \mathbf{r}_{\mathrm{CM}}\right) \dot{\boldsymbol{\omega}}+m\left(\mathbf{r}_{\mathrm{CM}} \cdot \dot{\boldsymbol{\omega}}\right) \mathbf{r}_{\mathrm{CM}}=-\boldsymbol{\omega} \times(\mathbf{I} \cdot \boldsymbol{\omega})-\mathbf{I}^{\prime} \cdot \boldsymbol{\omega}+m\left(\boldsymbol{\omega} \cdot \mathbf{r}_{\mathrm{CM}}\right) \mathbf{r}_{\mathrm{CM}} \times \boldsymbol{\omega}+ \\
   & m \mathbf{r}_{\mathrm{CM}} \times \mathbf{r}_{\mathrm{CM}}^{\prime \prime}+2 m \mathbf{r}_{\mathrm{CM}} \times \boldsymbol{\omega} \times \mathbf{r}_{\mathrm{CM}}^{\prime}-\mathbf{r}_{\mathrm{CM}} \times \mathbf{T}+2 \dot{m} \mathbf{r}_{\mathrm{CM}} \times \mathbf{r}_{\mathrm{CM}}^{\prime}-2 \mathbf{r}_{\mathrm{CM}} \times \boldsymbol{\omega} \times \dot{m}\left(\mathbf{r}_{\mathrm{noz}}-\mathbf{r}_{\mathrm{CM}}\right)- \\
   & \ddot{m} \mathbf{r}_{\mathrm{CM}} \times\left(\mathbf{r}_{\mathrm{noz}}-\mathbf{r}_{\mathrm{CM}}\right)-\mathbf{r}_{\mathrm{CM}} \times \mathbf{A}-\mathbf{r}_{\mathrm{CM}} \times \sum_i \mathbf{N}_i+\left(\dot{m} \mathbf{S}_{\mathrm{noz}}\right) \cdot \boldsymbol{\omega}+\sum_i \mathbf{r}_i \times \mathbf{N}_i
   \end{aligned}


**Solving for 𝐯̇:**

*Linear*

.. math::

   \begin{gathered}
   \dot{\mathbf{v}}=\frac{\left(\mathbf{T}-2 \dot{m} \mathbf{r}_{\mathrm{CM}}^{\prime}+2 \boldsymbol{\omega} \times \dot{m}\left(\mathbf{r}_{\mathrm{noz}}-\mathbf{r}_{\mathrm{CM}}\right)+\ddot{m}\left(\mathbf{r}_{\mathrm{noz}}-\mathbf{r}_{\mathrm{CM}}\right) \mathbf{A}+\sum_i \mathbf{N}_i\right)}{m}-g \hat{a}_3-\dot{\boldsymbol{\omega}} \times \mathbf{r}_{\mathrm{CM}} \\
   -\boldsymbol{\omega} \times\left(\boldsymbol{\omega} \times \mathbf{r}_{\mathrm{CM}}\right)-\mathbf{r}_{\mathrm{CM}}^{\prime \prime}-2 \boldsymbol{\omega} \times \mathbf{r}_{\mathrm{CM}}^{\prime}
   \end{gathered}

*Angular*

.. math::

   \mathbf{r}_{\mathrm{CM}} \times \dot{\mathbf{v}}=\frac{\left(\left(\dot{m} \mathbf{S}_{\mathrm{noz}}\right) \cdot \boldsymbol{\omega}+\sum_i \mathbf{r}_i \times \mathbf{N}_i-\mathbf{I} \cdot \dot{\boldsymbol{\omega}}-\boldsymbol{\omega} \times(\mathbf{I} \cdot \boldsymbol{\omega})-\mathbf{I}^{\prime} \cdot \boldsymbol{\omega}\right)}{\boldsymbol{m}}-\mathbf{r}_{\mathrm{CM}} \times g \hat{\mathbf{a}}_3

.. math::

   \mathbf{b}=\frac{\left(\left(\dot{m} \mathbf{S}_{\mathrm{noz}}\right) \cdot \boldsymbol{\omega}+\sum_i \mathbf{r}_i \times \mathbf{N}_i-\mathbf{I} \cdot \dot{\boldsymbol{\omega}}-\boldsymbol{\omega} \times(\mathbf{I} \cdot \boldsymbol{\omega})-\mathbf{I}^{\prime} \cdot \boldsymbol{\omega}\right)}{\boldsymbol{m}}-\mathbf{r}_{\mathrm{CM}} \times g \hat{\mathbf{a}}_3

.. math::

   \mathbf{r}_{\mathrm{CM}} \times \dot{\mathbf{v}}=\boldsymbol{b}


**Reorganized EOM to aid execution speed**

*Linear*

.. math::

   \begin{aligned}
   m \dot{\mathbf{v}}+\left[m \mathbf{r}_{\mathrm{CM}}\right]_{\times}^T & \cdot \dot{\boldsymbol{\omega}} \\
   & =-\omega \times\left(\omega \times m \mathbf{r}_{\mathrm{CM}}\right)+\omega \times\left(2 \dot{m}\left(\mathbf{r}_{\mathrm{noz}}-\mathbf{r}_{\mathrm{CM}}\right)-2 m \mathbf{r}_{\mathrm{CM}}^{\prime}\right)+\mathrm{T}-m \mathbf{r}_{\mathrm{CM}}^{\prime \prime}-2 \dot{m} \mathbf{r}_{\mathrm{CM}}^{\prime} \\
   & +\ddot{m}\left(\mathbf{r}_{\mathrm{noz}}-\mathbf{r}_{\mathrm{CM}}\right)-m g \hat{\mathbf{a}}_3+\mathbf{A}+\sum_i \mathbf{N}_i
   \end{aligned}

*Angular*

.. math::

   \mathbf{I} \cdot \dot{\boldsymbol{\omega}}+\left[m \mathbf{r}_{\mathrm{CM}}\right]_{\times} \cdot \dot{\mathbf{v}}=-\omega \times(\mathbf{I} \cdot \omega)+\left(\dot{m} \mathbf{S}_{\mathrm{noz}}-\mathbf{I}^{\prime}\right) \cdot \omega-\mathbf{r}_{\mathrm{CM}} \times m g \hat{\mathbf{a}}_3+\sum_i \mathbf{r}_i \times \mathbf{N}_i



**Available terms that must be interpolated in time/altitude**

1. :math:`m`: mass
2. :math:`𝑚'`: time derivative of :math:`m`
3. :math:`𝑚''`: time derivative of :math:`𝑚'`
4. :math:`r_{CM}`:
5. :math:`r_{CM}'`:
6. :math:`r_{CM}''`: 
7. :math:`T`: thrust
8. :math:`I`: inertia tensor
9. :math:`I'`: time derivative of :math:`I`
10. :math:`g`: gravity acceleration

**Pre-computed terms that optimize interpolations needed**
 
1. :math:`m`: mass
2. :math:`\mathrm{r}_{CM}`: position vector of the center of mass
3. :math:`\mathbf{T}_{03}`: :math:`2\dot{m} \left( r_{noz} - r_{CM} \right) - 2 \cdot m \cdot r_{CM}`
4. :math:`\mathbf{T}_{04}`: :math:`T - m \cdot r_{CM}' - 2 \cdot 𝑚̇ \cdot r_{CM} + 𝑚̈ \cdot (r_{noz} - r_{CM})`
5. :math:`\mathbf{T}_{05}`: :math:`\dot{m} \cdot S_{noz} - I'`
6. :math:`g`: gravity acceleration
7. :math:`\mathbf{I}`: inertia tensor
 
Pre-computed terms

1. :math:`\mathbf{T}_{00}`: :math:`m \cdot \mathrm{r}_{\mathrm{CM}}`
2. :math:`\mathbf{T}_{01}`: :math:`[m \cdot \mathrm{r}_{\mathrm{CM}}] \times`
3. :math:`\mathbf{T}_{02}`: :math:`[m \cdot \mathrm{r}_{\mathrm{CM}}] \times \mathbf{T}'`
4. :math:`\mathbf{T}_{03}`: :math:`2\cdot \dot{m} (\mathrm{r}_{noz} - \mathrm{r}_{\mathrm{CM}}) - 2 \cdot m \mathrm{r}_{\mathrm{CM}}`
5. :math:`\mathbf{T}_{04}`: :math:`\mathbf{T} - m \cdot \mathrm{r}_{\mathrm{CM}}'' - 2 \cdot \dot{m} \cdot \mathrm{r}_{\mathrm{CM}} + \ddot{m} (\mathrm{r}_{noz} - \mathrm{r}_{\mathrm{CM}})`
6. :math:`\mathbf{T}_{05}`: :math:`\dot{m} \cdot S_{noz} - \mathbf{I}'`
7. :math:`\mathbf{T}_{20}`: :math:`-\omega \times (\omega \times \mathbf{T}_{00}) + \omega \times (\mathbf{T}_{03}) + \mathbf{T}_{04} - m \cdot g \hat{a}_3 + \mathbf{A} + \sum \mathbf{N}_{i}`
8. :math:`\mathbf{T}_{21}`: :math:`-\omega \times (\mathbf{I} \cdot \omega) + (T_{05}) \cdot \omega + \mathrm{r}_{\mathrm{CM}} \times m \cdot g \hat{a}_3 + \sum r_{i} \times \mathbf{N}_{i}`

**Final system of equations**

.. math::
   
   \mathrm{M} \cdot \dot{\mathbf{v}}+\left[m \mathbf{r}_{\mathrm{CM}}\right]_{\times}^T \cdot \dot{\boldsymbol{\omega}}=T_{20}
   
   \mathbf{I} \cdot \dot{\boldsymbol{\omega}}+\left[\mathrm{mr}_{\mathrm{CM}}\right]_x \cdot \dot{\mathbf{v}}=T_{21}

**Solution to system of equations**

.. math::
   \dot{\boldsymbol{\omega}}=\left(\left(\mathrm{I}-\left[m \mathbf{r}_{\mathrm{CM}}\right]_X \cdot \mathrm{M}^{-1} \cdot\left[m \mathbf{r}_{\mathrm{CM}}\right]_X^T\right)\right)^{-1} \cdot\left(T_{21}-\left[m \mathbf{r}_{\mathrm{CM}}\right]_X \cdot \mathrm{M}^{-1} \cdot T_{20}\right)

.. math::
   \dot{\mathbf{v}}=\mathrm{M}^{-1} \cdot\left(T_{20}-\left[m \mathbf{r}_{\mathrm{CM}}\right]_{\times}^T \cdot \dot{\boldsymbol{\omega}}\right)

**Taking a closer look at the matrix inversion:**

.. math::

   \mathbf{H}=\left[m \mathbf{r}_{\mathrm{CM}}\right]_{\times} \cdot \mathbf{M}^{-1} \cdot\left[m \mathbf{r}_{\mathrm{CM}}\right]_{\times}^T

.. math::

   \mathbf{H}=-m\left[\mathrm{r}_{\mathrm{CM}}\right]_{\times}^2

.. math::

   \mathbf{H}=-m\left[\begin{array}{ccc}
   0 & -r_{\mathrm{CM}_3} & r_{\mathrm{CM}_2} \\
   r_{\mathrm{CM}_3} & 0 & -r_{\mathrm{CM}_1} \\
   -r_{\mathrm{CM}_2} & r_{\mathrm{CM}_1} & 0
   \end{array}\right]^2

.. math::

   \mathbf{H} = -m \left[ \begin{array}{ccc}
   0 & -r_{\mathrm{CM}_3} & r_{\mathrm{CM}_2} \\
   r_{\mathrm{CM}_3} & 0 & -r_{\mathrm{CM}_1} \\
   -r_{\mathrm{CM}_2} & r_{\mathrm{CM}_1} & 0
   \end{array}\right]\left[\begin{array}{ccc}
   0 & -r_{\mathrm{CM}_3} & r_{\mathrm{CM}_2} \\
   r_{\mathrm{CM}_3} & 0 & -r_{\mathrm{CM}_1} \\
   -r_{\mathrm{CM}_2} & r_{\mathrm{CM}_1} & 0
   \end{array} \right]

.. math::

   \mathbf{H}=-m\left[\begin{array}{ccc}
   -r_{\mathrm{CM}_3}^2-r_{\mathrm{CM}_2}^2 & r_{\mathrm{CM}_2} r_{\mathrm{CM}_1} & r_{\mathrm{CM}_3} r_{\mathrm{CM}_1} \\
   r_{\mathrm{CM}_2} r_{\mathrm{CM}_1} & -r_{\mathrm{CM}_3}^2-r_{\mathrm{CM}_1^2} & r_{\mathrm{CM}_3} r_{\mathrm{CM}_2} \\
   r_{\mathrm{CM}_3} r_{\mathrm{CM}_1} & r_{\mathrm{CM}_3} r_{\mathrm{CM}_2} & -r_{\mathrm{CM}_2}-r_{\mathrm{CM}_1}^2
   \end{array}\right]

.. math::

   \mathbf{H} = m \left[ \begin{array}{ccc}
   r_{\mathrm{CM}_3}^2+r_{\mathrm{CM}_2}^2 & -r_{\mathrm{CM}_2} r_{\mathrm{CM}_1} & -r_{\mathrm{CM}_3} r_{\mathrm{CM}_1} \\
   -r_{\mathrm{CM}_2} r_{\mathrm{CM}_1} & r_{\mathrm{CM}_3}^2+r_{\mathrm{CM}_1}^2 & -r_{\mathrm{CM}_3} r_{\mathrm{CM}_2} \\
   -r_{\mathrm{CM}_3} r_{\mathrm{CM}_1} & -r_{\mathrm{CM}_3} r_{\mathrm{CM}_2} & r_{\mathrm{CM}_2}^2+r_{\mathrm{CM}_1}^2
   \end{array}\right]


**Consider** :math:`I_{CM}` **as the inertia tensor relative to the true center of mass. Then:**

.. math::

   \mathbf{I}_{\mathrm{CM}}+\mathbf{H}=\mathbf{I}

   \mathbf{I}_{\mathrm{CM}}=\mathbf{I}-\mathbf{H}


**New simplified equations:**

.. math::

   \dot{\omega} = \mathbf{I}_{\mathrm{CM}}{ }^{-1} \cdot\left(T_{21}-\left[\mathrm{r}_{\mathrm{CM}}\right]_{\times} \cdot T_{20} \right)

   \dot{\mathbf{v}} = \mathrm{M}^{-1} \cdot \left( T_{20}-\left[m \mathrm{r}_{\mathrm{CM}}\right]_{\mathrm{x}}^T \cdot \dot{\omega} \right)

