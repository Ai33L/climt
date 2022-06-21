from sympl import initialize_numpy_arrays_with_properties, get_constant
from sympl import Stepper
import numpy as np
import numba
from numba import jit


@jit(nopython=True, parallel=True)
def Parallel_flux(air_temperature, surface_temperature, air_pressure,
                  air_pressure_int, surface_pressure, specific_humidity,
                  surface_humidity, northward_wind, eastward_wind,
                  new_air_temperature, new_specific_humidity,
                  new_northward_wind, new_eastward_wind,
                  Sensible_diag, Latent_diag, North_stress_diag,
                  East_stress_diag, Rd_val, Cp_val, g_val, k_val,
                  z0_val, P0_val, Ric_val, L, num_cols, timestep):

    Rd = Rd_val
    Cp_dry = Cp_val
    g = g_val
    k = k_val
    z0 = z0_val
    P0 = P0_val
    Ri_c = Ric_val

    for col in numba.prange(num_cols):

        air_temp = air_temperature[0, col]
        surf_temp = surface_temperature[col]
        air_press = air_pressure[0, col]
        air_press_int = air_pressure_int[:, col]
        surf_press = surface_pressure[col]
        spec_hum = specific_humidity[0, col]
        spec_hum_surf = surface_humidity[col]
        north_wind = northward_wind[0, col]
        east_wind = eastward_wind[0, col]

        wind = np.sqrt(np.power(north_wind, 2)+np.power(east_wind, 2))
        if wind < 1:
            wind = 1
        rho = air_press/(Rd * (1+0.608 * spec_hum) * air_temp)
        pot_temp = air_temp * np.power((P0/air_press), Rd/Cp_dry)
        pot_temp_surf = surf_temp * np.power((P0/surf_press), Rd/Cp_dry)

        virt_temp = pot_temp*(1+0.608*spec_hum)
        virt_temp_surf = pot_temp_surf*(1+0.608*spec_hum_surf)
        z = (Rd*air_temp*(1+0.608*spec_hum)/g) * np.log(surf_press/air_press)
        layer_thickness = (air_press_int[0]-air_press_int[1])/g
        Ri = g*z*(virt_temp-virt_temp_surf)/(virt_temp_surf*wind*wind)

        C = 0
        if Ri < 0:
            C = k*k*np.power(np.log(z/z0), -2)
        elif Ri < Ri_c:
            C = k*k*np.power(np.log(z/z0), -2)*np.power((1-Ri/Ri_c), 2)

        sat_vapour = 0.611*np.exp(17.3*surf_temp/(surf_temp+237.3))
        sat_spec_hum = 0.622*sat_vapour/surf_press

        North_wind_stress = rho*C*wind*north_wind
        East_wind_stress = rho*C*wind*east_wind
        Sensible_flux = rho*Cp_dry*C*wind*(pot_temp-pot_temp_surf)
        Evaporation = rho*C*wind*(spec_hum-sat_spec_hum)

        Sensible_diag[col] = -Sensible_flux
        Latent_diag[col] = -L*Evaporation
        North_stress_diag[col] = -North_wind_stress
        East_stress_diag[col] = -East_wind_stress

        new_air_temperature[0, col] = air_temp-Sensible_flux\
            / (Cp_dry*layer_thickness) * timestep
        new_specific_humidity[0, col] = spec_hum-Evaporation\
            / (layer_thickness) * timestep
        new_northward_wind[0, col] = north_wind-North_wind_stress\
            / (layer_thickness) * timestep
        new_eastward_wind[0, col] = east_wind-East_wind_stress\
            / (layer_thickness) * timestep


class SimpleFlux(Stepper):
    """
    This is a simple flux component that calculates the sensible and latent
    heat flux and surface stress. This component also makes changes to the
    temperature, humidity and wind values at the lowest model level as
    prescribed by the fluxes.

    This component is meant to be run before the SimpleBoundaryLayer component,
    as the modified values at the lowest level is crucial for the boundary
    layer to work.
    """

    input_properties = {
        'air_temperature': {
            'dims': ['mid_levels', '*'],
            'units': 'degK ',
        },
        'specific_humidity': {
            'dims': ['mid_levels', '*'],
            'units': 'kg/kg',
        },
        'air_pressure': {
            'dims': ['mid_levels', '*'],
            'units': 'Pa',
        },
        'air_pressure_on_interface_levels': {
            'dims': ['interface_levels', '*'],
            'units': 'Pa',
        },
        'northward_wind': {
            'dims': ['mid_levels', '*'],
            'units': 'm s^-1',
        },
        'eastward_wind': {
            'dims': ['mid_levels', '*'],
            'units': 'm s^-1',
        },
        'surface_air_pressure': {
            'dims': ['*'],
            'units': 'Pa',
        },
        'surface_temperature': {
            'dims': ['*'],
            'units': 'degK',
        },
        'surface_specific_humidity': {
            'dims': ['*'],
            'units': 'kg/kg',
        },
    }

    output_properties = {
        'air_temperature': {
            'dims': ['mid_levels', '*'],
            'units': 'degK ',
        },
        'specific_humidity': {
            'dims': ['mid_levels', '*'],
            'units': 'kg/kg',
        },
        'northward_wind': {
            'dims': ['mid_levels', '*'],
            'units': 'm s^-1',
        },
        'eastward_wind': {
            'dims': ['mid_levels', '*'],
            'units': 'm s^-1',
        },
    }

    diagnostic_properties = {
        'surface_upward_sensible_heat_flux': {
            'dims': ['*'],
            'units': 'W m^-2',
        },
        'surface_upward_latent_heat_flux': {
            'dims': ['*'],
            'units': 'W m^-2',
        },
        'northward_surface_stress': {
            'dims': ['*'],
            'units': 'Pa',
        },
        'eastward_surface_stress': {
            'dims': ['*'],
            'units': 'Pa',
        },
    }

    def __init__(self, von_karman_constant=0.4, roughness_length=0.0000321,
                 reference_pressure=100000, critical_richardson_number=1,
                 **kwargs):
        """
        Args:
        roughness_length:
            A measure of the surface roughness.
        reference_pressure:
            The reference pressure used in the potential temperature
            calculations.
        critical_richardson_number:
            A set threshold value which determines the diffusion coefficients
            and the height of the boundary layer.
        """

        self._k = von_karman_constant
        self._z0 = roughness_length
        self._P0 = reference_pressure
        self._Ric = critical_richardson_number
        self._update_constants()

        super(SimpleFlux, self).__init__(**kwargs)

    def _update_constants(self):

        self._Rd = get_constant('gas_constant_of_dry_air', 'J kg^-1 K^-1')
        self._Cp =\
            get_constant('heat_capacity_of_dry_air_at_constant_pressure',
                         'J kg^-1 K^-1')
        self._g = get_constant('gravitational_acceleration', 'm s^-2')
        self._L =\
            get_constant('latent_heat_of_vaporization_of_water', 'J kg^-1')

    def array_call(self, state, timestep):
        """
        Calculates the surface stress, sensible flux and latent flux for each
        column. The lowest level values of temperature, humidity and wind are
        changed according to the flux.
        """

        num_cols = state['air_temperature'].shape[1]

        new_state = initialize_numpy_arrays_with_properties(
            self.output_properties, state, self.input_properties
        )

        new_state['air_temperature'][:] = state["air_temperature"]
        new_state['specific_humidity'][:] = state['specific_humidity']
        new_state['northward_wind'][:] = state['northward_wind']
        new_state['eastward_wind'][:] = state['eastward_wind']

        diagnostics = initialize_numpy_arrays_with_properties(
            self.diagnostic_properties, state, self.input_properties
        )

        Parallel_flux(state['air_temperature'],
                      state['surface_temperature'],
                      state['air_pressure'],
                      state['air_pressure_on_interface_levels'],
                      state['surface_air_pressure'],
                      state['specific_humidity'],
                      state['surface_specific_humidity'],
                      state['northward_wind'], state['eastward_wind'],
                      new_state['air_temperature'],
                      new_state['specific_humidity'],
                      new_state['northward_wind'],
                      new_state['eastward_wind'],
                      diagnostics['surface_upward_sensible_heat_flux'],
                      diagnostics['surface_upward_latent_heat_flux'],
                      diagnostics['northward_surface_stress'],
                      diagnostics['eastward_surface_stress'],
                      self._Rd, self._Cp, self._g, self._k, self._z0,
                      self._P0, self._Ric, self._L, num_cols,
                      timestep.total_seconds())

        return diagnostics, new_state
