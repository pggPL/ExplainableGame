#!/bin/bash

if [ "$#" -lt 1 ]; then
    echo "Usage: ./run.sh <command> [options]"
    echo "Commands:"
    echo "  train - trains the model"
    echo "  play - play a game against the trained model"
    echo "  server - start the web server to play the game in a browser"
    exit 1
fi

command=$1
shift

case $command in
    train)
        echo "Training the model..."
        python train.py "$@"
        ;;

    play)
        echo "Playing a game against the trained model..."
        python play.py "$@"
        ;;

    server)
        echo "Starting the web server..."
        python app.py "$@"
        ;;

    *)
        echo "Invalid command. Use 'train', 'play', or 'server'."
        exit 1
        ;;
esac
