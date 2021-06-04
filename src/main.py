import asyncio

from src.network import Network
from aiostream import stream, pipe, operator


async def main():
    network = Network(2, 1)

    data = [
        [[0, 0], [0]],
        [[0, 1], [1]],
        [[1, 0], [1]],
        [[1, 1], [0]],
    ]

    @operator
    def train(data_set, epochs=10000):
        for i in range(0, epochs):
            network.train_once(data_set)

    @operator(pipable=True)
    def print_result(t_d):
        for (i, index) in t_d:
            network.input(i)
            print(f"{i[0]} XOR {i[1]} = {network.prediction}")

    test_data = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]

    xs = (
      train(data)
      | print_result.pipe(test_data)
    )
    await xs


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
