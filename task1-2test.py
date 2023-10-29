from data.data import get_coords
from main import predict_traffic_flow
import argparse


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scats",
        default="4034",
        help="SCATS site number.")
    parser.add_argument(
        "--direction",
        default="E",
        help="The approach to the site (N, S, E, W, NE, NW, SE)")
    parser.add_argument(
        "--time",
        default="13:30",
        help="The time in 24 hr notation")
    parser.add_argument(
        "--date",
        default="1/10/2006",
        help="The day of the month")
    parser.add_argument(
        "--model",
        default="lstm",
        help="Model to use for prediction (lstm, gru, saes)")
    args = parser.parse_args()
    
    lat, long = get_coords('data/Scats Data October 2006.csv', args.scats, args.direction)
    if (lat == -1):
        print(args.direction + " is not a valid direction for " + args.scats)
    else:
        flow_prediction = predict_traffic_flow(lat, long, args.date, args.time, args.model)
        print(flow_prediction)

