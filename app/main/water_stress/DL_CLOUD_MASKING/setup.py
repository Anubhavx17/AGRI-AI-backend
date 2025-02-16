from setuptools import setup, find_packages
# import app.main.water_stress.DL_CLOUD_MASKING.dl_l8s2_uv as utils

setup(name='dl_l8s2_uv',
      version='0.1',
      description='Cloud masking of Landsat-8 and Sentinel-2 based on Deep Learning',
      author='Dan Lopez Puigdollers',
      author_email='dan.lopez@uv.es',
      packages=find_packages(exclude=["tests"]),
      package_data={'': ['*.hdf5']},
      include_package_data=True,
      install_requires=["numpy", "lxml", "spectral", "luigi", "h5py", "rasterio", "shapely"],
      zip_safe=False)

