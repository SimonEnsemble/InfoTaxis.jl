import marimo

__generated_with = "0.11.23"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.special import kn # 	Modified Bessel function of the second kind
    from dataclasses import dataclass
    import pymc as pm
    import pytensor
    import arviz as az
    import scipy
    return az, dataclass, kn, mo, np, pd, plt, pm, pytensor, scipy


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# plume model ::fluent-emoji-flat:pig-nose::""")
    return


@app.cell
def _(dataclass, np):
    @dataclass
    class Plume:
        # source strength
        r: float
        # diffusion coeff
        D : float
        # life span
        tau: float
        # wind velocity
        v : np.array
        # location
        x_0 : np.array

        def __init__(self, r, D, tau, v, x_0):
            self.r = r
            self.D = D
            self.tau = tau
            self.v = v
            self.x_0 = x_0
            self.kappa = np.sqrt((np.dot(v, v) + 4 * D / tau) / (4 * D ** 2))
    return (Plume,)


@app.cell
def _(kn):
    kn(0, 2.0)
    return


@app.cell
def _(Plume):
    def new_plume(plume, x_0, r):
        return Plume(
            x_0=x_0, r=r, D=plume.D, tau=plume.tau, v=plume.v
        )
    return (new_plume,)


@app.cell
def _(Plume, np):
    plume = Plume(
        v = np.array([-5.0, 15.0]), # m/s
    	D = 25.0, # m²/min
    	tau = 50.0, # min
    	x_0 = np.array([25.0, 4.0]), # m
    	r = 10.0 # g/min
    )
    return (plume,)


@app.cell
def _(kn, np):
    def c(x, plume):
        dx = x - plume.x_0
        d = np.linalg.norm(dx)
        return plume.r / (2 * np.pi * plume.D) * kn(0, plume.kappa * d) * np.exp(
            np.dot(plume.v, dx) / (2 * plume.D)
        )
    return (c,)


@app.cell
def _(c, np, plt):
    def viz_plume(plume, L, dx=0.05):
        x = np.arange(0, L, dx)
        y = np.arange(0, L, dx)

        X, Y = np.meshgrid(x, y)

        Z = [[c(np.array([x_i, y_j]), plume) for x_i in x] for y_j in y]

        plt.pcolormesh(x, y, Z, cmap='viridis', vmin=0.0, vmax=0.1)
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.colorbar(label="concentration")
        plt.show()
    return (viz_plume,)


@app.cell
def _(c, plume):
    c([30,30], plume)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## measurement model""")
    return


@app.cell
def _(plume, sample_c_obs):
    sample_c_obs([30, 20], plume, 0.01)
    return


@app.cell
def _(c, np):
    def sample_c_obs(x, plume, sigma):
        return c(x, plume) + np.random.randn() * sigma
    return (sample_c_obs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# InfoTaxis runs""")
    return


@app.cell
def _(Plume, dataclass):
    @dataclass
    class InfoTaxisRun:
        plume: Plume
        sigma: float
        dx: float
        L: float
        actions: list

        def __init__(self, plume, sigma, dx, L):
            self.plume = plume
            self.sigma = sigma
            self.dx = dx
            self.L = L
            self.actions = [[0.0, dx], [0.0, -dx], [dx, 0.0], [-dx, 0.0]]

        def get_action_id(self, action):
            # hard coded. check self.actions.
            if action == "up":
                return 0
            elif action == "down":
                return 1
            elif action == "right":
                return 2
            elif action == "left":
                return 3
    return (InfoTaxisRun,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# robot moving and taking measurements ::fluent-emoji-flat:robot::""")
    return


@app.cell
def _(InfoTaxisRun, plume):
    settings = InfoTaxisRun(
        plume=plume, sigma=0.005, dx=1.0, L=35
    )
    return (settings,)


@app.cell
def _(settings, viz_plume):
    viz_plume(settings.plume, settings.L, dx=settings.dx)
    return


@app.cell
def _(np, pd, sample_c_obs, settings):
    class Robot:
        def __init__(self, x, settings):
            self.settings = settings

            # measure concentration
            c_obs = sample_c_obs(x, settings.plume, settings.sigma)

            # initialize data
            self.data = self.collect_data(x)

        def collect_data(self, x):
            c_obs = sample_c_obs(x, self.settings.plume, self.settings.sigma)
            return pd.DataFrame(
                {
                    'x': [x],
                    'c_obs': [c_obs]
                }
            )

        def move(self, action_id, n=1):
            for i in range(n):
                last_id = self.data.index[-1]
                x = self.data.loc[last_id, "x"] 
                dx = self.settings.actions[action_id]
                x_new = x + dx

                if np.any(x_new < 0.0) or np.any(x_new > settings.L):
                    raise Exception("moving outside box")

                new_data = self.collect_data(x_new)
                self.data = pd.concat((self.data, new_data), ignore_index=True)
    return (Robot,)


@app.cell
def _(Robot, np, settings):
    robot = Robot(
        # initial position
        np.array([0.0, 0.0]), 
        # environment and sensor info
        settings
    )

    robot.move(settings.get_action_id("up"), n=3)
    robot.move(settings.get_action_id("right"), n=5)
    # robot.move(settings.get_action_id("up"), n=3)
    # robot.move(settings.get_action_id("right"), n=7)
    # robot.move(settings.get_action_id("up"), n=8)

    robot.data
    return (robot,)


@app.cell
def _(plt):
    def viz_robot(robot, settings):
        plt.figure()
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")

        plt.xlim(-settings.dx, settings.L + settings.dx)
        plt.ylim(-settings.dx, settings.L + settings.dx)

        # bounding box
        plt.plot([0, settings.L], [0, 0], color="black")
        plt.plot([0, settings.L], [settings.L, settings.L], color="black")
        plt.plot([0, 0], [0, settings.L], color="black")
        plt.plot([settings.L, settings.L], [0, settings.L], color="black")

        # data
        plt.scatter(
            [x[0] for x in robot.data["x"]],
            [x[1] for x in robot.data["x"]],
            c=robot.data["c_obs"].values
        )
        plt.colorbar(label="c_obs")

        plt.show()
    return (viz_robot,)


@app.cell
def _(robot, settings, viz_robot):
    viz_robot(robot, settings)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 🕵️‍♂️ Bayesian inference of the plume source and strength""")
    return


@app.cell
def _(np, pytensor):
    # gotta use pytensor here so autodiff works
    def c_pymc(x, plume):
        dx = x - plume.x_0
        d = pytensor.tensor.linalg.norm(dx)
        return plume.r / (2 * np.pi * plume.D) * pytensor.tensor.kv(
            0, plume.kappa * d) * np.exp(
            pytensor.tensor.dot(plume.v, dx) / (2 * plume.D)
        )
    return (c_pymc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""test that `c_pymc` and `c` match.""")
    return


@app.cell
def _(c, c_pymc, np, plume, pytensor):
    # test that c_pymc and c match
    # evaluate tensor expression
    _x_test = [20.0, 15]
    _expr = c_pymc(_x_test, plume)
    _f = pytensor.function([], _expr)
    _c_pymc = _f()

    _c_reg = float(c(_x_test, plume))

    assert np.isclose(_c_reg, _c_pymc)
    return


@app.cell
def _(c_pymc, new_plume, np, plume, pm, robot, settings):
    forward_model = pm.Model()

    with forward_model:
        # prior over source location and strength
        X_0 = pm.Uniform(
            "x_0", 
            lower=np.array([0, 0]), 
            upper=np.array([settings.L, settings.L]), 
            shape=(2, )
        )
        R = pm.Uniform("r", lower=0.0, upper=50.0)

        # create plume with this source
        modified_plume = new_plume(plume, x_0=X_0, r=R)

        # expected value of outcome
        mu = [c_pymc(x, modified_plume) for x in robot.data["x"].values]

        # Likelihood (sampling distribution) of observations
        C_obs = pm.Normal(
            "c_obs", 
            mu=mu, sigma=2*settings.sigma, 
            observed=robot.data["c_obs"].values
        )
    return C_obs, R, X_0, forward_model, modified_plume, mu


@app.cell
def _(forward_model, pm):
    with forward_model:
        # draw 1000 posterior samples
        idata = pm.sample(draws=100)
    return (idata,)


@app.cell
def _(idata):
    idata.posterior
    return


@app.cell
def _(az, idata):
    az.plot_forest(
        idata, combined=False, hdi_prob=0.90
    )
    return


@app.cell
def _(az, idata, plt, settings):
    _ax = az.plot_kde(
        idata.posterior["x_0"][:, :, 0],
        idata.posterior["x_0"][:, :, 1],
        contour_kwargs={"colors": None, "cmap": plt.cm.viridis, "levels": 30},
        contourf_kwargs={"alpha": 0.5, "levels": 30},
    )
    _ax.set_xlim(0, settings.L)
    _ax.set_ylim(0, settings.L)
    _ax.set_title("NUTS")
    _ax
    return


@app.cell
def _(forward_model, pm):
    with forward_model:
        svgd_approx = pm.fit(
            100,
            method="svgd",
            inf_kwargs=dict(n_particles=100)
            # obj_optimizer=pm.sgd(learning_rate=0.01),
        )
    return (svgd_approx,)


@app.cell
def _(svgd_approx):
    svgd_posterior = svgd_approx.sample(100)
    return (svgd_posterior,)


@app.cell
def _(az, plt, settings, svgd_posterior):
    _ax = az.plot_kde(
        svgd_posterior.posterior["x_0"][:, :, 0],
        svgd_posterior.posterior["x_0"][:, :, 1],
        contour_kwargs={"colors": None, "cmap": plt.cm.viridis, "levels": 30},
        contourf_kwargs={"alpha": 0.5, "levels": 30},
    )
    _ax.set_xlim(0, settings.L)
    _ax.set_ylim(0, settings.L)
    _ax.set_title("VI")
    _ax
    return


@app.cell
def _(idata):
    idata.posterior["x_0"].values
    return


@app.cell
def _(az, idata, plt):
    az.plot_kde(idata.posterior["r"].values,
                contour_kwargs={"colors":None, "cmap":plt.cm.viridis},
                contourf_kwargs={"alpha":0})
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
