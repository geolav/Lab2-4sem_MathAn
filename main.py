import numpy as np
import matplotlib.pyplot as plt


def f_original(x):
    if 0 <= x < 1:
        return 2 * x
    elif 1 <= x <= 2:
        return 1
    return 0


# 1. Общий ряд Фурье (Период T = 2, l = 1)
l_gen = 1
x_int_gen = np.linspace(0, 2, 5000)
# Сама исходная функция на периоде [0, 2]
f_vals_gen = np.where(x_int_gen < 1, 2 * x_int_gen, 1)

a0_gen = np.trapezoid(f_vals_gen, x_int_gen) / l_gen


def a_general(n):
    integral = f_vals_gen * np.cos(np.pi * n * x_int_gen / l_gen)
    return np.trapezoid(integral, x_int_gen) / l_gen


def b_general(n):
    integral = f_vals_gen * np.sin(np.pi * n * x_int_gen / l_gen)
    return np.trapezoid(integral, x_int_gen) / l_gen


def general_sum(x, N):
    s = np.full_like(x, a0_gen / 2, dtype=float)
    for n in range(1, N + 1):
        s += a_general(n) * np.cos(np.pi * n * x / l_gen)
        s += b_general(n) * np.sin(np.pi * n * x / l_gen)
    return s


# 2. Косинусный и Синусный ряды (Период T = 4, L = 2)
L_trig = 2
x_int_trig = np.linspace(0, 2, 5000)
f_vals_trig = np.where(x_int_trig < 1, 2 * x_int_trig, 1)


def a_cos(n):
    integral = f_vals_trig * np.cos(np.pi * n * x_int_trig / L_trig)
    return (2 / L_trig) * np.trapezoid(integral, x_int_trig)

def cos_sum(x, N):
    s = np.full_like(x, a_cos(0) / 2, dtype=float)
    for n in range(1, N + 1):
        s += a_cos(n) * np.cos(np.pi * n * x / L_trig)
    return s


def b_sin(n):
    integral = f_vals_trig * np.sin(np.pi * n * x_int_trig / L_trig)
    return (2 / L_trig) * np.trapezoid(integral, x_int_trig)

def sin_sum(x, N):
    s = np.zeros_like(x)
    for n in range(1, N + 1):
        s += b_sin(n) * np.sin(np.pi * n * x / L_trig)
    return s


def plot_original():
    x1 = np.linspace(0, 1, 500, endpoint=False)
    x2 = np.linspace(1, 2, 500, endpoint=False)
    plt.plot(x1, 2 * x1, linewidth=3, color='blue', label='Исходная функция')
    plt.plot(x2, np.ones_like(x2), linewidth=3, color='blue')


def plot_periodic():
    for k in range(-2, 3):
        shift = 2 * k
        x1 = np.linspace(0 + shift, 1 + shift, 300, endpoint=False)
        x2 = np.linspace(1 + shift, 2 + shift, 300, endpoint=False)
        plt.plot(x1, 2 * (x1 - shift), linewidth=3, color='blue',
                 label='Периодический оригинал' if k == -2 else "")
        plt.plot(x2, np.ones_like(x2), linewidth=3, color='blue')


def plot_even_extension():
    # Симметрия относительно Y (Период = 4)
    for k in range(-1, 2):
        shift = 4 * k
        x1 = np.linspace(-2 + shift, -1 + shift, 300, endpoint=False)
        x2 = np.linspace(-1 + shift, 0 + shift, 300, endpoint=False)
        x3 = np.linspace(0 + shift, 1 + shift, 300, endpoint=False)
        x4 = np.linspace(1 + shift, 2 + shift, 300, endpoint=False)

        plt.plot(x1, np.ones_like(x1), color='blue', linewidth=3, label='Чётное продолжение' if k == -1 else "")
        plt.plot(x2, -2 * x2 + 2 * shift, color='blue', linewidth=3)
        plt.plot(x3, 2 * x3 - 2 * shift, color='blue', linewidth=3)
        plt.plot(x4, np.ones_like(x4), color='blue', linewidth=3)


def plot_odd_extension():
    # Центральная симметрия (Период = 4)
    for k in range(-1, 2):
        shift = 4 * k
        x1 = np.linspace(-2 + shift, -1 + shift, 300, endpoint=False)
        x2 = np.linspace(-1 + shift, 0 + shift, 300, endpoint=False)
        x3 = np.linspace(0 + shift, 1 + shift, 300, endpoint=False)
        x4 = np.linspace(1 + shift, 2 + shift, 300, endpoint=False)

        plt.plot(x1, -np.ones_like(x1), color='blue', linewidth=3, label='Нечётное продолжение' if k == -1 else "")
        plt.plot(x2, 2 * x2 - 2 * shift, color='blue', linewidth=3)
        plt.plot(x3, 2 * x3 - 2 * shift, color='blue', linewidth=3)
        plt.plot(x4, np.ones_like(x4), color='blue', linewidth=3)



def make_plot(x, y_func, title, filename, mode="original"):
    plt.figure(figsize=(10, 6))
    if mode == "general":
        plot_periodic()
    elif mode == "even":
        plot_even_extension()
    elif mode == "odd":
        plot_odd_extension()
    else:
        plot_original()

    for N in [4, 10, 40, 100]:
        plt.plot(x, y_func(x, N), label=f'N={N}')
    plt.grid()
    plt.xlim(x[0], x[-1])
    plt.legend()
    plt.title(title)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


x_global = np.linspace(-4, 4, 4000)
x_zoom = np.linspace(0, 2, 1000)

make_plot(x_global, general_sum, "Общий ряд Фурье (общий вид)", "general_global.png", mode="general")
make_plot(x_zoom, general_sum, "Общий ряд Фурье (увеличенный вид)", "general_zoom.png", mode="general")

make_plot(x_global, cos_sum, "Косинусный ряд (общий вид)", "cos_global.png", mode="even")
make_plot(x_zoom, cos_sum, "Косинусный ряд (увеличенный вид)", "cos_zoom.png", mode="even")

make_plot(x_global, sin_sum, "Синусный ряд (общий вид)", "sin_global.png", mode="odd")
make_plot(x_zoom, sin_sum, "Синусный ряд (увеличенный вид)", "sin_zoom.png", mode="odd")

print("Все графики успешно обновлены и сохранены.")