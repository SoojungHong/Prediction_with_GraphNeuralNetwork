# libraries
from math import sin, cos, sqrt, atan2, radians


# functions
def measure_distance_in_km(query_lat, query_lon, result_lat, result_lon):
    # approximate radius of earth in km
    R = 6373.0

    if not query_lat.empty:
        q_lat = radians(query_lat)
        q_lon = radians(query_lon)
        r_lat = radians(result_lat)
        r_lon = radians(result_lon)

        dlon = r_lon - q_lon
        dlat = r_lat - q_lat

        a = sin(dlat / 2) ** 2 + cos(q_lat) * cos(r_lat) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance = R * c
        # print("Result in km:", distance)
    else:
        print("query_lat is empty")
        distance = 0

    return distance

def measure_distance(query_lat, query_lon, result_lat, result_lon):
    # approximate radius of earth in km
    R = 6373.0

    if not query_lat.empty:
        q_lat = radians(query_lat)
        q_lon = radians(query_lon)
        r_lat = radians(result_lat)
        r_lon = radians(result_lon)

        dlon = r_lon - q_lon
        dlat = r_lat - q_lat

        a = sin(dlat / 2) ** 2 + cos(q_lat) * cos(r_lat) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance = R * c
        # print("Result in km:", distance)
    else:
        print("query_lat is empty")
        distance = 0

    return distance