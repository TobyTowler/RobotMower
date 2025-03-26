import fields2cover as f2c


def headlandGen():
    rand = f2c.Random(42)
    field = rand.generateRandField(1e4, 5)
    cells = field.getField()
    robot = f2c.Robot(2.0, 26.0)
    const_hl = f2c.HG_Const_gen()
    no_hl = const_hl.generateHeadlands(cells, 3.0 * robot.getWidth())
    print(
        "The complete area is ",
        cells.area(),
        ", and the area without headlands is ",
        no_hl.area(),
    )
    return field, no_hl


def drawCell(field, headland):
    f2c.Visualizer.figure()
    # for i in cell:
    f2c.Visualizer.plot(field)
    f2c.Visualizer.plot(headland)
    f2c.Visualizer.show()
    # f2c.Visualizer.save("Tutorial_image.pn


field, hl = headlandGen()
drawCell(field, hl)
