======================
Equations of Motion v1
======================

Introduction
============

This document briefly describes the equations of motion which govern the flight
of a rocket used in v1.0 onwards. This document simply shows some of the 
algebraic steps used to get to the final form of the equations of motion used
in the code. For a more detailed explanation of the equations of motion, please
refer to :ref:`Equations of Motion v0 <eqsv0>`.

Development
-----------

**Linear Equation:**

.. math:: 
   \begin{aligned}
   m\left(\dot{\mathbf{v}}+\dot{\boldsymbol{\omega}} \times \mathbf{r}_{\mathrm{CM}}\right. & \left.+\boldsymbol{\omega} \times\left(\boldsymbol{\omega} \times \mathbf{r}_{\mathrm{CM}}\right)+\mathbf{r}_{\mathrm{CM}}^{\prime \prime}+2 \boldsymbol{\omega} \times \mathbf{r}_{\mathrm{CM}}^{\prime}\right) \\
   = & \mathbf{T}-2 \dot{m} \mathbf{r}_{\mathrm{CM}}^{\prime}+2 \boldsymbol{\omega} \times \dot{m}\left(\mathbf{r}_{\mathrm{noz}}-\mathbf{r}_{\mathrm{CM}}\right)+\ddot{m}\left(\mathbf{r}_{\mathrm{noz}}-\mathbf{r}_{\mathrm{CM}}\right)+\mathbf{A}+\sum_i \mathbf{N}_i-m g \hat{\mathbf{a}}_3
   \end{aligned}


**Angular Equation:**

.. math:: \mathbf{I} \cdot \dot{\boldsymbol{\omega}}+\boldsymbol{\omega} \times(\mathbf{I} \cdot \boldsymbol{\omega})+\mathbf{I}^{\prime} \cdot \boldsymbol{\omega}+m \mathbf{r}_{\mathrm{CM}} \times \dot{\mathbf{v}}=\left(\dot{m} \mathbf{S}_{\mathrm{noz}}\right) \cdot \boldsymbol{\omega}+\sum_i \mathbf{r}_i \times \mathbf{N}_i-\mathbf{r}_{\mathrm{CM}} \times m g \hat{\mathbf{a}}_3

**Cross multiplying the linear equation by 𝐫CM:**

.. math:: 
   \begin{aligned}
   m \mathbf{r}_{\mathrm{CM}} \times \dot{\mathbf{v}}+m \mathbf{r}_{\mathrm{CM}} \times\left(\dot{\boldsymbol{\omega}} \times \mathbf{r}_{\mathrm{CM}}\right)+m \mathbf{r}_{\mathrm{CM}} \times \boldsymbol{\omega} \times\left(\boldsymbol{\omega} \times \mathbf{r}_{\mathrm{CM}}\right)+m \mathbf{r}_{\mathrm{CM}} \times \mathbf{r}_{\mathrm{CM}}^{\prime \prime}+2 m \mathbf{r}_{\mathrm{CM}} \times \boldsymbol{\omega} \times \mathbf{r}_{\mathrm{CM}}^{\prime} \\
   \quad=\mathbf{r}_{\mathrm{CM}} \times \mathbf{T}-2 \dot{m} \mathbf{r}_{\mathrm{CM}} \times \mathbf{r}_{\mathrm{CM}}^{\prime}+2 \mathbf{r}_{\mathrm{CM}} \times \boldsymbol{\omega} \times \dot{m}\left(\mathbf{r}_{\mathrm{noz}}-\mathbf{r}_{\mathrm{CM}}\right)+m \mathbf{r}_{\mathrm{CM}}^{\prime} \times\left(\mathbf{r}_{\mathrm{noz}}-\mathbf{r}_{\mathrm{CM}}\right) \\
   \quad+\mathbf{r}_{\mathrm{CM}} \times \mathbf{A}+\mathbf{r}_{\mathrm{CM}} \times \sum_i \mathbf{N}_i-m \mathbf{r}_{\mathrm{CM}} \times g \hat{\mathbf{a}}_3
   \end{aligned}

**Simplifying:**

.. math:: \mathbf{r}_{\mathrm{CM}} \times\left(\dot{\omega} \times \mathbf{r}_{\mathrm{CM}}\right)=\left(\mathbf{r}_{\mathrm{CM}} \cdot \mathbf{r}_{\mathrm{CM}}\right) \dot{\boldsymbol{\omega}}-\left(\mathbf{r}_{\mathrm{CM}} \cdot \dot{\boldsymbol{\omega}}\right) \mathbf{r}_{\mathrm{CM}}

.. math:: \mathbf{r}_{\mathrm{CM}} \times \boldsymbol{\omega} \times\left(\boldsymbol{\omega} \times \mathbf{r}_{\mathrm{CM}}\right)=\mathbf{r}_{\mathrm{CM}} \times\left(\left(\boldsymbol{\omega} \cdot \mathbf{r}_{\mathrm{CM}}\right) \boldsymbol{\omega}-(\boldsymbol{\omega} \cdot \boldsymbol{\omega}) \mathbf{r}_{\mathrm{CM}}\right)=\left(\boldsymbol{\omega} \cdot \mathbf{r}_{\mathrm{CM}}\right) \mathbf{r}_{\mathrm{CM}} \times \boldsymbol{\omega}

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

1. 𝑚
2. 𝑚'
3. 𝑚''
4. 𝐫CM
5. 𝐫CM'
6. 𝐫CM''
7. 𝐓
8. 𝐈
9. 𝐈'
10. 𝑔

**Pre-computed terms that optimize interpolations needed**
 
1. 𝑚
2. 𝐫CM'
3. T03: 2𝑚̇ (𝐫noz − 𝐫CM) − 2𝑚𝐫CM
4. T04: 𝐓 − 𝑚𝐫CM′′ − 2𝑚̇ 𝐫CM + 𝑚̈ (𝐫noz − 𝐫CM)
5. T05: 𝑚̇ 𝐒noz − 𝐈′
6. 𝑔
7. 𝐈
 
Pre-computed terms

1. T00: 𝑚𝐫CM
2. T01: [𝑚𝐫CM]×
3. T02: [𝑚𝐫CM]×𝑇′
4. T03: 2𝑚̇ (𝐫noz − 𝐫CM) − 2𝑚𝐫CM
5. T04: 𝐓 − 𝑚𝐫CM′′ − 2𝑚̇ 𝐫CM + 𝑚̈ (𝐫noz − 𝐫CM)
6. T05: 𝑚̇ 𝐒noz − 𝐈′
7. T20: −𝝎 × (𝝎 × 𝑇00) + 𝝎 × (𝑇03) + 𝑇04 − 𝑚𝑔𝐚̂3 + 𝐀 + ∑ 𝐍𝑖
8. T21: −𝝎 × (𝐈 ⋅ 𝝎) + (𝑇05) ⋅ 𝝎 + 𝐫CM × 𝑚𝑔𝐚̂3 + ∑ 𝐫𝑖 × 𝐍𝑖 

**Final system of equations**

- .. math:: \mathrm{M} \cdot \dot{\mathbf{v}}+\left[m \mathbf{r}_{\mathrm{CM}}\right]_{\times}^T \cdot \dot{\boldsymbol{\omega}}=T_{20}
- .. math:: \mathbf{I} \cdot \dot{\boldsymbol{\omega}}+\left[\mathrm{mr}_{\mathrm{CM}}\right]_x \cdot \dot{\mathbf{v}}=T_{21}

**Solution to system of equations**

.. math::
   \dot{\boldsymbol{\omega}}=\left(\left(\mathrm{I}-\left[m \mathbf{r}_{\mathrm{CM}}\right]_X \cdot \mathrm{M}^{-1} \cdot\left[m \mathbf{r}_{\mathrm{CM}}\right]_X^T\right)\right)^{-1} \cdot\left(T_{21}-\left[m \mathbf{r}_{\mathrm{CM}}\right]_X \cdot \mathrm{M}^{-1} \cdot T_{20}\right)

.. math::
   \dot{\mathbf{v}}=\mathrm{M}^{-1} \cdot\left(T_{20}-\left[m \mathbf{r}_{\mathrm{CM}}\right]_{\times}^T \cdot \dot{\boldsymbol{\omega}}\right)

**Taking a closer look at the matrix inversion:**

.. math::
   \begin{equation}
   \mathbf{H}=\left[m \mathbf{r}_{\mathrm{CM}}\right]_{\times} \cdot \mathbf{M}^{-1} \cdot\left[m \mathbf{r}_{\mathrm{CM}}\right]_{\times}^T
   \end{equation}

.. math::
   \begin{equation}
   \mathbf{H}=-m\left[\mathrm{r}_{\mathrm{CM}}\right]_{\times}^2
   \end{equation}

.. math::
   \begin{equation}
   \mathbf{H}=-m\left[\begin{array}{ccc}
   0 & -r_{\mathrm{CM}_3} & r_{\mathrm{CM}_2} \\
   r_{\mathrm{CM}_3} & 0 & -r_{\mathrm{CM}_1} \\
   -r_{\mathrm{CM}_2} & r_{\mathrm{CM}_1} & 0
   \end{array}\right]^2
   \end{equation}

.. math::
   \begin{equation}
   \mathbf{H}=-m\left[\begin{array}{ccc}
   0 & -r_{\mathrm{CM}_3} & r_{\mathrm{CM}_2} \\
   r_{\mathrm{CM}_3} & 0 & -r_{\mathrm{CM}_1} \\
   -r_{\mathrm{CM}_2} & r_{\mathrm{CM}_1} & 0
   \end{array}\right]\left[\begin{array}{ccc}
   0 & -r_{\mathrm{CM}_3} & r_{\mathrm{CM}_2} \\
   r_{\mathrm{CM}_3} & 0 & -r_{\mathrm{CM}_1} \\
   -r_{\mathrm{CM}_2} & r_{\mathrm{CM}_1} & 0
   \end{array}\right]
   \end{equation}

.. math::
   \begin{equation}
   \mathbf{H}=-m\left[\begin{array}{ccc}
   -r_{\mathrm{CM}_3}^2-r_{\mathrm{CM}_2}^2 & r_{\mathrm{CM}_2} r_{\mathrm{CM}_1} & r_{\mathrm{CM}_3} r_{\mathrm{CM}_1} \\
   r_{\mathrm{CM}_2} r_{\mathrm{CM}_1} & -r_{\mathrm{CM}_3}^2-r_{\mathrm{CM}_1^2} & r_{\mathrm{CM}_3} r_{\mathrm{CM}_2} \\
   r_{\mathrm{CM}_3} r_{\mathrm{CM}_1} & r_{\mathrm{CM}_3} r_{\mathrm{CM}_2} & -r_{\mathrm{CM}_2}-r_{\mathrm{CM}_1}^2
   \end{array}\right]
   \end{equation}

.. math::
   \begin{equation}
   \mathbf{H}=m\left[\begin{array}{ccc}
   r_{\mathrm{CM}_3}^2+r_{\mathrm{CM}_2}^2 & -r_{\mathrm{CM}_2} r_{\mathrm{CM}_1} & -r_{\mathrm{CM}_3} r_{\mathrm{CM}_1} \\
   -r_{\mathrm{CM}_2} r_{\mathrm{CM}_1} & r_{\mathrm{CM}_3}^2+r_{\mathrm{CM}_1}^2 & -r_{\mathrm{CM}_3} r_{\mathrm{CM}_2} \\
   -r_{\mathrm{CM}_3} r_{\mathrm{CM}_1} & -r_{\mathrm{CM}_3} r_{\mathrm{CM}_2} & r_{\mathrm{CM}_2}^2+r_{\mathrm{CM}_1}^2
   \end{array}\right]
   \end{equation}

**Consider 𝐼CM as the inertia tensor relative to the true center of mass. Then:**

.. math::
   \begin{equation}
   \mathbf{I}_{\mathrm{CM}}+\mathbf{H}=\mathbf{I}
   \end{equation}

.. math::
   \begin{equation}
   \mathbf{I}_{\mathrm{CM}}=\mathbf{I}-\mathbf{H}
   \end{equation}

**New simplified equations:**

.. math::
   \begin{equation}
   \dot{\omega}=\mathbf{I}_{\mathrm{CM}}{ }^{-1} \cdot\left(T_{21}-\left[\mathrm{r}_{\mathrm{CM}}\right]_{\times} \cdot T_{20}\right)
   \end{equation}

.. math::
   \begin{equation}
   \dot{\mathbf{v}}=\mathrm{M}^{-1} \cdot\left(T_{20}-\left[m \mathrm{r}_{\mathrm{CM}}\right]_{\mathrm{x}}^T \cdot \dot{\boldsymbol{\omega}}\right)
   \end{equation}