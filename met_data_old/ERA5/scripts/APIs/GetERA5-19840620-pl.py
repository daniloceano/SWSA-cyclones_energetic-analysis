import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type':'reanalysis',
        'format':'netcdf',
        'pressure_level':[
            '1','2','3',
            '5','7','10',
            '20','30','50',
            '70','100','125',
            '150','175','200',
            '225','250','300',
            '350','400','450',
            '500','550','600',
            '650','700','750',
            '775','800','825',
            '850','875','900',
            '925','950','975',
            '1000'
        ],
        'date':'19840919/19840924',
        'area':'-27.760126/-61.099701/-47.842861/-33.49054',
        'time':'00/to/23/by/1',
        'variable':[
            'geopotential','vertical_velocity',
            'temperature','u_component_of_wind','v_component_of_wind'
        ]
    },
    '19840620_ERA5.nc')
