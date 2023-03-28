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
        'date':'19830702/19830710',
        'area':'-26.576305/-60.682739/-55.781956/-17.630066',
        'time':'00/to/23/by/1',
        'variable':[
            'geopotential','vertical_velocity',
            'temperature','u_component_of_wind','v_component_of_wind'
        ]
    },
    '19830422_ERA5.nc')
