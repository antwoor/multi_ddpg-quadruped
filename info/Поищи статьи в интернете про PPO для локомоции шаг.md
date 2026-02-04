<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Поищи статьи в интернете про PPO для локомоции шагабщего четырёхногого робота и выдай мне лучшую функцию вознаграждения

Для одной «лучшей» функции вознаграждения в литературе консенсус примерно такой: нужно одновременно стимулировать трекинг скорости, устойчивость позы и энергоэффективность, с сильным штрафом за падение. Ниже — собранный, «усреднённый» вариант из современных работ по PPO-локомоции квадрупеда (Unitree/ANYmal‑подобные), который можно спокойно вставлять в статью как основной.

***

## Рекомендуемая функция вознаграждения для PPO-локомоции квадрупеда

Обозначения на шаге $t$:

- $v_x, v_y$ — фактические линейные скорости корпуса по осям $x,y$.
- $v_x^{\text{cmd}}, v_y^{\text{cmd}}$ — заданные (командные) скорости.
- $\omega_z$ — фактическая угловая скорость рыскания, $\omega_z^{\text{cmd}}$ — заданная.
- $(\phi,\theta,\psi)$ — углы крена, тангажа, рыскания.
- $h$, $h^{\text{ref}}$ — текущая и опорная высота корпуса.
- $\tau_i$ — крутящий момент в $i$-м суставе, $q_i$ и $q_i^{\text{ref}}$ — фактический и «референтный» (нейтральный) угол сустава.
- $c_i \in \{0,1\}$ — контакт ступни $i$ с опорой.
- $\eta$ — запас устойчивости (расстояние проекции CoM до границы опорного многоугольника).
- $\mathbf{1}_{\text{fall}}$ — индикатор падения/терминации эпизода.

Тогда удобная общая форма:

$$
r_t = 
r_{\text{vel}} +
r_{\text{orient}} +
r_{\text{height}} +
r_{\text{smooth}} +
r_{\text{stance}} +
r_{\text{stability}} +
r_{\text{survive}} +
r_{\text{term}} .
$$

Развернём компоненты:

1. **Слежение за командой по скорости**

$$
r_{\text{vel}} = 
k_v^{\text{lin}}\left[
-\big(v_x - v_x^{\text{cmd}}\big)^2
-\big(v_y - v_y^{\text{cmd}}\big)^2
\right]
+ 
k_v^{\text{ang}}\left[
-\big(\omega_z - \omega_z^{\text{cmd}}\big)^2
\right].
$$

Это заставляет робота реализовывать желаемую поступательную и поворотную скорость.
2. **Стабильная ориентация корпуса**

$$
r_{\text{orient}} = 
-k_{\text{post}}\big(\phi^2 + \theta^2\big),
$$

рыскание $\psi$ обычно не штрафуют напрямую, если оно входит в команду; крен и тангаж должны быть малы.
3. **Контроль высоты корпуса**

$$
r_{\text{height}} = 
-k_h\big(h - h^{\text{ref}}\big)^2.
$$
4. **Энергоэффективность и плавность движений**
    - штраф за моменты:

$$
r_{\text{torque}} = 
-k_\tau \sum_i \tau_i^2;
$$
    - штраф за «дёрганость» (разность действий по времени):

$$
r_{\text{smooth}} = 
-k_{\Delta a} \sum_i (a_{t,i} - a_{t-1,i})^2.
$$
5. **Контакт и паттерн шага** (простая версия)

$$
r_{\text{stance}} = 
k_{\text{contact}} \left(
\frac{1}{N_\text{feet}}\sum_i c_i
\right),
$$

поощряет наличие опорных ног (робот не «летит» в воздухе постоянно). В продвинутых версиях сюда добавляют целевые фазы/частоты шага, но для статьи можно оставить простой контакт‑терм.
6. **Устойчивость по опорному многоугольнику**

$$
r_{\text{stability}} = k_{\eta}\,\eta,
$$

растёт, если CoM хорошо «внутри» опорного многоугольника.
7. **Выживание и падение**

$$
r_{\text{survive}} = k_{\text{survive}},\quad
r_{\text{term}} = -k_{\text{fall}}\cdot \mathbf{1}_{\text{fall}}.
$$

Малое постоянно положительное вознаграждение за каждый шаг и большой разовый штраф за падение.

Итог (в компактной записи, как удобно вставить в раздел «Reward function»):

$$
\begin{aligned}
r_t =&
-k_v^{\text{lin}}\left[
(v_x - v_x^{\text{cmd}})^2
+ (v_y - v_y^{\text{cmd}})^2
\right]
-k_v^{\text{ang}}(\omega_z - \omega_z^{\text{cmd}})^2 \\
&-k_{\text{post}}(\phi^2 + \theta^2)
-k_h(h - h^{\text{ref}})^2
-k_\tau \sum_i \tau_i^2
-k_{\Delta a} \sum_i (a_{t,i} - a_{t-1,i})^2 \\
&+k_{\text{contact}}\left(
\frac{1}{N_\text{feet}}\sum_i c_i
\right)
+k_{\eta}\,\eta
+k_{\text{survive}}
-k_{\text{fall}}\cdot \mathbf{1}_{\text{fall}}.
\end{aligned}
$$

Такую структуру (с акцентом на velocity tracking + posture + energy + smoothness + survival/punish-fall) используют практически во всех современный PPO‑локомоционных работах с четвероногими (Unitree A1/Go1, ANYmal и т.п.), меняя только конкретные веса и детали «стойко‑поощрительных» терминов.

Если хочешь, я могу:

- Подобрать пример чисел для коэффициентов $k_\cdot$ (под скорость 0.5–1 м/с и шаг 0.01–0.02 c).
- Переписать это коротким английским текстом прямо под раздел статьи (формат «We design the reward as a weighted sum of …»).
<span style="display:none">[^1][^2][^3][^4][^5][^6][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: PZ_VKR_last-4.pdf

[^2]: https://arxiv.org/html/2403.01928v1

[^3]: https://arxiv.org/html/2601.20668v1

[^4]: https://www.youtube.com/watch?v=X6R8S499dXg

[^5]: http://evgiz.net/article/2019/02/10/

[^6]: https://arxiv.org/pdf/2504.09242.pdf

[^7]: https://www.youtube.com/watch?v=2Epn__SRHns

[^8]: https://www.reddit.com/r/reinforcementlearning/comments/1m76ps1/quadruped_locomotion_with_ppo_how_to_move_forward/

[^9]: https://habr.com/ru/companies/otus/articles/888122/

