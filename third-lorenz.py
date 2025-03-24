from manim import *
from manim.opengl import *
from scipy.interpolate import interp1d
import numpy as np

class LorenzAttractor(ThreeDScene):
    def construct(self):
        self.create_lorenz()

    def lorenz_system(self, pos, sigma=10, rho=28, beta=8/3):
        x, y, z = pos
        dxdt = sigma * (y - x)
        dydt = x * (rho - z) - y
        dzdt = x * y - beta * z
        return np.array([dxdt, dydt, dzdt])

    def rate_to_color(self, rate, min_rate, max_rate):
        epsilon = 1e-6
        if min_rate == max_rate: 
            return BLUE

        rate_log = np.log(rate + epsilon)
        min_rate_log = np.log(min_rate + epsilon)
        max_rate_log = np.log(max_rate + epsilon)
        rate_normalized = (rate_log - min_rate_log) / (max_rate_log - min_rate_log)
        return interpolate_color(BLUE, PURPLE, rate_normalized)

    def create_lorenz(self):
        num_curves = 12
        all_curves = VGroup() 
        colors = [BLUE_A, BLUE_B, BLUE_C, BLUE_D, GREEN_A, GREEN_B, RED_A, RED_B, BLUE_A, BLUE_E, GREEN_E, BLUE_A]

        initial_position = np.array([-1.0, 1.0, 0.0]) 
        
        for curve_idx in range(num_curves):
            trajectory = []
            pos = initial_position.copy()  
            dt = 0.001
            steps = 40000
            scale_factor = 0.1

            min_rate = float("inf")
            max_rate = float("-inf")

            # Generate trajectory for each curve
            for _ in range(steps):
                dp = self.lorenz_system(pos) * dt
                rate = np.linalg.norm(dp)
                min_rate = min(min_rate, rate)
                max_rate = max(max_rate, rate)
                pos += dp
                trajectory.append(pos.copy())

            trajectory = np.array(trajectory)

            # Interpolation function for smooth curve rendering
            interp_func = interp1d(
                np.linspace(0, 1, steps), trajectory, axis=0, kind="linear", fill_value="extrapolate"
            )

            curve = ParametricFunction(
                lambda t: interp_func(np.clip(t, 0, 1)) * scale_factor,
                t_range=[0, 1, 0.002]
            )

            curve.move_to(ORIGIN) 
            curve.set_stroke(width=2)
            curve.set_color_by_gradient(RED, BLUE)
            curve.set_color(colors[curve_idx % len(colors)]) 
            all_curves.add(curve) 
       
        self.set_camera_orientation(phi=80 * DEGREES, theta=45 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.8)

        lorenz_equation = MathTex(r"\begin{cases} \dot{x} = \sigma(y-x) \\ \dot{y} = x(\rho - z) - y \\ \dot{z} = xy - \beta z \end{cases}").scale(0.8)
        lorenz_equation.to_edge(UL)

        try:
            self.add_fixed_in_frame_mobjects(lorenz_equation)
        except AttributeError:
            self.add(lorenz_equation)

        self.play(Write(lorenz_equation), run_time=2)
        self.play(LaggedStartMap(Create, all_curves, lag_ratio=0.2, run_time=35))
        self.wait()
