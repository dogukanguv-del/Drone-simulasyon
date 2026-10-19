"""
Quadrotor birleşik kinematik + dinamik model + kaskad PID kontrolcü (TEK DOSYA).

Koordinat düzeni:
  - Dünya çerçevesi: z YUKARI pozitif (yerçekimi -z yönünde)
  - Gövde çerçevesi: itki +z_gövde yönünde
  - Euler açıları: Z-Y-X (psi -> theta -> phi), radyan

Durum vektörü (12 eleman):
  [x, y, z,          -> konum (dünya, m)
   phi, theta, psi,  -> roll, pitch, yaw (rad)
   u, v, w,          -> doğrusal hız (GÖVDE, m/s)
   p, q, r]          -> açısal hız (gövde, rad/s)

Giriş: 4 motor hızı [w1, w2, w3, w4] (rad/s), artı (+) dizilim.
"""
import numpy as np

# --- FİZİKSEL PARAMETRELER ---
m = 1.0        # kütle (kg)
g = 9.81       # yerçekimi (m/s^2)
L = 0.25       # kol uzunluğu (m)
k = 3e-6       # itki katsayısı  (N / (rad/s)^2)
b = 1e-7       # tork katsayısı  (N.m / (rad/s)^2)
I = np.diag([0.0081, 0.0081, 0.0142])   # atalet matrisi (kg.m^2)
I_inv = np.linalg.inv(I)

W_MIN, W_MAX = 0.0, 1500.0              # motor hız sınırları (rad/s)
W_HOVER = np.sqrt(m * g / (4 * k))      # hover motor hızı ≈ 904.2 rad/s


# ---------------------------------------------------------------
# YARDIMCI: Rotasyon matrisi (gövde -> dünya)
# ---------------------------------------------------------------
def rotation_matrix(phi, theta, psi):
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    c_theta, s_theta = np.cos(theta), np.sin(theta)
    c_psi, s_psi = np.cos(psi), np.sin(psi)
    return np.array([
        [c_theta*c_psi, s_phi*s_theta*c_psi - c_phi*s_psi, c_phi*s_theta*c_psi + s_phi*s_psi],
        [c_theta*s_psi, s_phi*s_theta*s_psi + c_phi*c_psi, c_phi*s_theta*s_psi - s_phi*c_psi],
        [-s_theta,      s_phi*c_theta,                     c_phi*c_theta]
    ])


# ---------------------------------------------------------------
# 1) KİNEMATİK: konum ve açı değişim hızları
# ---------------------------------------------------------------
def euler_kinematics(angles, v_body, omega_body):
    """
    angles: [phi, theta, psi]
    v_body: [u, v, w]  gövde çerçevesinde doğrusal hız
    omega_body: [p, q, r]
    Dönüş: pos_dot (dünya, 3), ang_dot (Euler açı türevleri, 3)
    """
    phi, theta, psi = angles
    p, q, r = omega_body

    c_phi, s_phi = np.cos(phi), np.sin(phi)
    c_theta = np.cos(theta)
    t_theta = np.tan(theta)

    # Konum türevi: R * v_body
    pos_dot = rotation_matrix(phi, theta, psi) @ np.asarray(v_body)

    # Gimbal lock: theta = ±90° civarında T matrisi tekildir
    if abs(c_theta) < 1e-3:
        print("Uyarı: Gimbal Lock (tekillik) sınırı!")
        c_theta = 1e-3 if c_theta >= 0 else -1e-3
        t_theta = np.sin(theta) / c_theta

    T = np.array([
        [1, s_phi * t_theta, c_phi * t_theta],
        [0, c_phi,          -s_phi],
        [0, s_phi / c_theta, c_phi / c_theta]
    ])
    ang_dot = T @ np.asarray(omega_body)
    return pos_dot, ang_dot


# ---------------------------------------------------------------
# 2) MİKSER: motor hızları -> toplam itki ve torklar
# ---------------------------------------------------------------
def mixer(motor_speeds):
    w = np.clip(np.asarray(motor_speeds, dtype=float), W_MIN, W_MAX)
    w_sq = w**2
    T_total = k * np.sum(w_sq)
    tau_phi = k * L * (w_sq[3] - w_sq[1])
    tau_theta = k * L * (w_sq[0] - w_sq[2])
    tau_psi = b * (w_sq[0] - w_sq[1] + w_sq[2] - w_sq[3])
    return T_total, np.array([tau_phi, tau_theta, tau_psi])


# ---------------------------------------------------------------
# 3) DİNAMİK: doğrusal ve açısal ivmeler (gövde çerçevesinde)
# ---------------------------------------------------------------
def body_dynamics(angles, v_body, omega_body, T_total, tau):
    """
    Newton (gövde çerçevesi):  m*(v_dot + ω x v) = F_itki + R^T * F_yerçekimi
    Euler:                     I*ω_dot + ω x (I*ω) = tau
    """
    phi, theta, psi = angles
    v_body = np.asarray(v_body)
    omega_body = np.asarray(omega_body)

    R = rotation_matrix(phi, theta, psi)

    thrust_body = np.array([0.0, 0.0, T_total])
    gravity_body = R.T @ np.array([0.0, 0.0, -m * g])   # yerçekimi gövde eksenine

    v_dot = (thrust_body + gravity_body) / m - np.cross(omega_body, v_body)
    omega_dot = I_inv @ (tau - np.cross(omega_body, I @ omega_body))
    return v_dot, omega_dot


# ---------------------------------------------------------------
# 4) BİRLEŞİK MODEL: durum türevi (12 eleman)
# ---------------------------------------------------------------
def drone_model(state, motor_speeds):
    pos = state[0:3]
    angles = state[3:6]
    v_body = state[6:9]
    omega_body = state[9:12]

    T_total, tau = mixer(motor_speeds)
    pos_dot, ang_dot = euler_kinematics(angles, v_body, omega_body)
    v_dot, omega_dot = body_dynamics(angles, v_body, omega_body, T_total, tau)

    return np.concatenate([pos_dot, ang_dot, v_dot, omega_dot])


# ---------------------------------------------------------------
# 5) İNTEGRASYON VE SİMÜLASYON
# ---------------------------------------------------------------
def rk4_step(state, motor_speeds, dt):
    k1 = drone_model(state, motor_speeds)
    k2 = drone_model(state + 0.5 * dt * k1, motor_speeds)
    k3 = drone_model(state + 0.5 * dt * k2, motor_speeds)
    k4 = drone_model(state + dt * k3, motor_speeds)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def wrap_angles(state):
    for i in (3, 5):   # roll ve yaw
        state[i] = (state[i] + np.pi) % (2 * np.pi) - np.pi
    return state


def simulate(initial_state, motor_fn, duration=2.0, dt=0.001):
    """motor_fn(t, state) -> 4 motor hızı. Dönüş: (t, durum geçmişi)."""
    n = int(duration / dt)
    t_hist = np.zeros(n + 1)
    s_hist = np.zeros((n + 1, 12))
    state = np.array(initial_state, dtype=float)
    s_hist[0] = state
    for i in range(n):
        t = i * dt
        state = wrap_angles(rk4_step(state, motor_fn(t, state), dt))
        t_hist[i + 1] = t + dt
        s_hist[i + 1] = state
    return t_hist, s_hist

import os
import sys
import matplotlib.pyplot as plt

# Bu dosyanın kendisi "dm" olarak kullanılır (dm.rotation_matrix, dm.rk4_step ...)
dm = sys.modules[__name__]


# ---------------------------------------------------------------
# PID SINIFI
# ---------------------------------------------------------------
class PID:
    """
    PID. Türev terimi dışarıdan (ölçülen hızdan) verilir; böylece hedef
    sıçramalarında 'derivative kick' oluşmaz. Koşullu integrasyonla anti-windup içerir.
    """
    def __init__(self, kp, ki=0.0, kd=0.0, out_min=-np.inf, out_max=np.inf, i_limit=np.inf):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self.i_limit = i_limit
        self.integral = 0.0

    def reset(self):
        self.integral = 0.0

    def update(self, error, error_rate, dt):
        out_unsat = self.kp * error + self.ki * self.integral + self.kd * error_rate
        out = np.clip(out_unsat, self.out_min, self.out_max)
        sat_high = out_unsat > self.out_max and error > 0
        sat_low = out_unsat < self.out_min and error < 0
        if not (sat_high or sat_low):
            self.integral += error * dt
            self.integral = np.clip(self.integral, -self.i_limit, self.i_limit)
        return out


# ---------------------------------------------------------------
# KONTROLCÜ
# ---------------------------------------------------------------
class DroneController:
    def __init__(self):
        # Dış döngü: konum (çıkış: yatay ivme, m/s^2)
        self.pid_x = PID(kp=2.25, ki=0.05, kd=2.7, out_min=-3, out_max=3, i_limit=2)
        self.pid_y = PID(kp=2.25, ki=0.05, kd=2.7, out_min=-3, out_max=3, i_limit=2)
        # İrtifa (çıkış: dikey ivme, m/s^2)
        self.pid_z = PID(kp=9.0, ki=0.2, kd=6.0, out_min=-6, out_max=8, i_limit=1)
        # İç döngü: açı (çıkış: açısal ivme, rad/s^2)
        self.pid_phi = PID(kp=100, kd=16, out_min=-200, out_max=200)
        self.pid_theta = PID(kp=100, kd=16, out_min=-200, out_max=200)
        self.pid_psi = PID(kp=25, kd=8, out_min=-100, out_max=100)

        self.max_tilt = np.radians(20)
        self.max_climb = 1.0     # m/s, irtifa hedefi rampa hızı
        self.z_cmd = 0.0

    def compute(self, state, setpoint, dt):
        """
        state: 12'li durum vektörü (drone_model formatı, hız GÖVDE çerçevesinde)
        setpoint: (x_ref, y_ref, z_ref, psi_ref)
        Dönüş: 4 motor hızı (rad/s)
        """
        x, y, z = state[0:3]
        angles = state[3:6]
        phi, theta, psi = angles
        v_body = state[6:9]
        omega_body = state[9:12]
        x_ref, y_ref, z_ref, psi_ref = setpoint

        # --- Ölçümleri kontrol için dönüştür ---
        R = dm.rotation_matrix(phi, theta, psi)
        v_world = R @ v_body                                   # gövde -> dünya hızı
        _, ang_dot = dm.euler_kinematics(angles, v_body, omega_body)
        phi_dot, theta_dot, psi_dot = ang_dot                  # gerçek Euler açı hızları

        # --- Konum -> istenen yatay ivme ---
        ax = self.pid_x.update(x_ref - x, -v_world[0], dt)
        ay = self.pid_y.update(y_ref - y, -v_world[1], dt)

        # Yaw'a göre döndürüp istenen roll/pitch'e çevir (küçük açı yaklaşımı)
        theta_des = (ax * np.cos(psi) + ay * np.sin(psi)) / g
        phi_des = (ax * np.sin(psi) - ay * np.cos(psi)) / g
        theta_des = np.clip(theta_des, -self.max_tilt, self.max_tilt)
        phi_des = np.clip(phi_des, -self.max_tilt, self.max_tilt)

        # --- İrtifa -> toplam itki (hız sınırlı rampa ile) ---
        step = np.clip(z_ref - self.z_cmd, -self.max_climb * dt, self.max_climb * dt)
        self.z_cmd += step
        az = self.pid_z.update(self.z_cmd - z, -v_world[2], dt)
        T = m * (g + az) / max(np.cos(phi) * np.cos(theta), 0.5)
        T = max(T, 0.0)

        # --- Açı -> torklar (tau = I * alfa) ---
        alpha_phi = self.pid_phi.update(phi_des - phi, -phi_dot, dt)
        alpha_theta = self.pid_theta.update(theta_des - theta, -theta_dot, dt)
        psi_err = (psi_ref - psi + np.pi) % (2 * np.pi) - np.pi
        alpha_psi = self.pid_psi.update(psi_err, -psi_dot, dt)

        tau_phi = I[0, 0] * alpha_phi
        tau_theta = I[1, 1] * alpha_theta
        tau_psi = I[2, 2] * alpha_psi

        return self.inverse_mixer(T, tau_phi, tau_theta, tau_psi)

    @staticmethod
    def inverse_mixer(T, tau_phi, tau_theta, tau_psi):
        """(itki, torklar) -> motor hızları. dm.mixer'ın tersi."""
        a = T / k
        bb = tau_phi / (k * L)
        c = tau_theta / (k * L)
        d = tau_psi / b

        w1_sq = (a + d) / 4 + c / 2
        w3_sq = (a + d) / 4 - c / 2
        w4_sq = (a - d) / 4 + bb / 2
        w2_sq = (a - d) / 4 - bb / 2

        w_sq = np.clip([w1_sq, w2_sq, w3_sq, w4_sq], 0.0, None)
        return np.clip(np.sqrt(w_sq), W_MIN, W_MAX)


# ---------------------------------------------------------------
# YER SINIRI (gövde çerçevesi hızına uygun)
# ---------------------------------------------------------------
def apply_ground(state):
    """z < 0 olamaz. Aşağı yönlü dünya hızı sıfırlanır, gövde hızına geri çevrilir."""
    if state[2] < 0.0:
        state[2] = 0.0
        R = dm.rotation_matrix(*state[3:6])
        v_world = R @ state[6:9]
        if v_world[2] < 0.0:
            v_world[2] = 0.0
            state[6:9] = R.T @ v_world
    return state


# ---------------------------------------------------------------
# SİMÜLASYON
# ---------------------------------------------------------------
def reference(t):
    """Zamana bağlı hedef: (x, y, z, yaw)."""
    z_ref = 2.0
    x_ref, y_ref = (0.0, 0.0) if t < 5 else (2.0, 1.0)
    psi_ref = 0.0 if t < 10 else np.radians(45)
    return x_ref, y_ref, z_ref, psi_ref


def run(duration=15.0, dt=0.002):
    ctrl = DroneController()
    n = int(duration / dt)
    state = np.zeros(12)

    t_hist = np.zeros(n + 1)
    s_hist = np.zeros((n + 1, 12))
    r_hist = np.zeros((n + 1, 4))
    w_hist = np.zeros((n + 1, 4))
    v_world_hist = np.zeros((n + 1, 3))
    s_hist[0] = state
    r_hist[0] = reference(0)

    for i in range(n):
        t = i * dt
        sp = reference(t)
        w = ctrl.compute(state, sp, dt)
        state = dm.rk4_step(state, w, dt)
        state = apply_ground(dm.wrap_angles(state))

        t_hist[i + 1] = t + dt
        s_hist[i + 1] = state
        r_hist[i + 1] = sp
        w_hist[i + 1] = w
        v_world_hist[i + 1] = dm.rotation_matrix(*state[3:6]) @ state[6:9]
    w_hist[0] = w_hist[1]
    return t_hist, s_hist, r_hist, w_hist, v_world_hist


def settle_time(t, y, ref, t0, tol):
    mask = t >= t0
    out = np.abs(y[mask] - ref) > tol
    if not out.any():
        return 0.0
    last = np.where(out)[0][-1]
    if last + 1 >= mask.sum():
        return np.nan
    return t[mask][last + 1] - t0


def plot(t, s, r, w, vw, path):
    fig, ax = plt.subplots(3, 2, figsize=(13, 10))
    ax = ax.ravel()

    ax[0].plot(t, r[:, 2], "--", label="z hedef")
    ax[0].plot(t, s[:, 2], label="z")
    ax[0].set_title("İrtifa (m)")
    ax[0].legend()

    ax[1].plot(t, r[:, 0], "--", color="tab:blue", label="x hedef")
    ax[1].plot(t, s[:, 0], color="tab:blue", label="x")
    ax[1].plot(t, r[:, 1], "--", color="tab:orange", label="y hedef")
    ax[1].plot(t, s[:, 1], color="tab:orange", label="y")
    ax[1].set_title("Yatay konum (m)")
    ax[1].legend()

    ax[2].plot(t, np.degrees(s[:, 3]), label="roll (φ)")
    ax[2].plot(t, np.degrees(s[:, 4]), label="pitch (θ)")
    ax[2].set_title("Roll / Pitch (derece)")
    ax[2].legend()

    ax[3].plot(t, np.degrees(r[:, 3]), "--", label="yaw hedef")
    ax[3].plot(t, np.degrees(s[:, 5]), label="yaw (ψ)")
    ax[3].set_title("Yaw (derece)")
    ax[3].legend()

    ax[4].plot(t, s[:, 6], label="u (gövde)")
    ax[4].plot(t, s[:, 7], label="v (gövde)")
    ax[4].plot(t, s[:, 8], label="w (gövde)")
    ax[4].plot(t, vw[:, 2], "k:", label="ż (dünya)")
    ax[4].set_title("Gövde hızları ve dünya dikey hızı (m/s)")
    ax[4].legend(ncol=2, fontsize=8)

    for j in range(4):
        ax[5].plot(t, w[:, j], label=f"motor {j+1}", linewidth=0.8)
    ax[5].set_title("Motor hızları (rad/s)")
    ax[5].legend(ncol=4, fontsize=8)

    for a in ax:
        a.set_xlabel("zaman (s)")
        a.grid(alpha=0.3)

    fig.suptitle("drone sonuçlar", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    t, s, r, w, vw = run()

    z_over = (s[:, 2].max() - 2.0) / 2.0 * 100
    z_settle = settle_time(t, s[:, 2], 2.0, 0.0, 0.05)
    xy_settle = max(settle_time(t, s[:, 0], 2.0, 5.0, 0.05),
                    settle_time(t, s[:, 1], 1.0, 5.0, 0.05))
    yaw_settle = settle_time(t, np.degrees(s[:, 5]), 45.0, 10.0, 1.0)

    print("=== SONUÇLAR ===")
    print(f"İrtifa aşımı           : %{z_over:.1f}")
    print(f"İrtifa oturma süresi   : {z_settle:.2f} s (±5 cm)")
    print(f"Yatay oturma süresi    : {xy_settle:.2f} s (±5 cm, 5. sn adımından sonra)")
    print(f"Yaw oturma süresi      : {yaw_settle:.2f} s (±1°, 10. sn adımından sonra)")
    print(f"Maks |roll|, |pitch|   : {np.degrees(np.abs(s[:,3]).max()):.1f}°, {np.degrees(np.abs(s[:,4]).max()):.1f}°")
    print(f"Son konum (x,y,z)      : {np.round(s[-1,:3], 3)}")
    print(f"Son yaw                : {np.degrees(s[-1,5]):.2f}°")
    print(f"Motor hız aralığı      : {w[1:].min():.0f} - {w[1:].max():.0f} rad/s")
    print(f"Min z (yer kontrolü)   : {s[:,2].min():.3f} m")

    plot(t, s, r, w, vw, os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_pid_sonuclar.png"))
    print("Grafik kaydedildi: model_pid_sonuclar.png")
    plt.show()
