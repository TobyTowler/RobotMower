import fields2cover as f2c


def drawCell(arr):
    f2c.Visualizer.figure()
    for i in arr:
        f2c.Visualizer.plot(i)
    f2c.Visualizer.show()


def main():
    rand = f2c.Random(42)
    field = rand.generateRandField(1e4, 5)
    cell = field.getField()
    drawCell(cell)


def squareField():
    cells = f2c.Cells(
        f2c.Cell(
            f2c.LinearRing(
                f2c.VectorPoint(
                    [
                        f2c.Point(0, 0),
                        f2c.Point(60, 0),
                        f2c.Point(60, 60),
                        f2c.Point(0, 60),
                        f2c.Point(0, 0),
                        # random.randint(0, 150),
                    ]
                )
            )
        )
    )
    drawCell(cells)


squareField()

main()
# Mapping.drawCell(cell)
