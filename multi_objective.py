import json
import math
import os
import random
import sys

from deap import base, creator, tools
from PIL import Image


class MyImage(object):
    """Class representing the source image. It preserves information about
    image *width*, *height*, list of *relevant_colors* and color values of some
    *pixels* according to given *step_size*.

    Relevant_colors: For each pixel, values for red, green and blue are
    rounded down to a multiple of *color_limitation_size*. Frequency of occurence of
    these rounded colors is calculated. Colors are then sorted according
    to their frequency and amount equal to *number_of_relevant_colors* of
    most frequent ones is selected.

    Pixels: Create a list of color values of pixels in an image. List
    contains a value for each pixel in every nth column of every nth row in
    an image where n equals *step_size*. Values for red, green and blue are
    rounded down to a multiple of *color_limitation_size*.
    """

    def __init__(self, name, color_limitation_size, number_of_relevant_colors, step_size):
        """Get properties of an image and create *self.relevant_colors* and
        *self.pixels* lists.

        :param name: Name of source image.
        :param color_limitation_size: Parameter for limiting color values.
        :param number_of_relevant_colors: Number of most frequent colors to be
                                          chosen to work with.
        :param step_size: Step size for skipping part of pixels.
        """
        with Image.open(name) as image:
            self.width, self.height = image.size
            frequency = {}
            for i in range(self.width):
                for j in range(self.height):
                    color = image.getpixel((i, j))
                    r = color[0] - (color[0] % color_limitation_size)
                    g = color[1] - (color[1] % color_limitation_size)
                    b = color[2] - (color[2] % color_limitation_size)
                    color_limited = (r, g, b)
                    frequency[color_limited] = frequency.get(color_limited, 0) + 1

            sorted_tuples = sorted(frequency.items(), key=lambda tup: tup[1], reverse=True)
            all_sorted_colors = [x for (x, y) in sorted_tuples]
            self.relevant_colors = all_sorted_colors[:number_of_relevant_colors]
            self.pixels = []
            for i in range(0, self.width, step_size):
                for j in range(0, self.height, step_size):
                    color = image.getpixel((i, j))[:3]
                    self.pixels.append(color)


class Picture(object):
    """Picture represents an individual. It contains a list of shapes
    *self.shapes*, method for saving the individual as SVG image *save*, five
    types of mutation (*mutate_color*, *mutate_dimensions*, *mutate_position*,
    *mutate_add* and *mutate_delete*) and two fitness functions
    (*evaluate_shapes_rgb* and *evaluate*).
    """

    def __init__(self, width, height, relevant_colors, number_of_objects):
        """Create object Picture. It contains *number_of_objects* items. This
        number has to be a multiple of four because of properties of
        TournamentDCD selection.

        :param width: Width of source image.
        :param height: Height of source image.
        :param relevant_colors: List of relevant colors according to frequency.
        :param number_of_object: Number of objects in one individual.
        """
        self.shapes = []
        for i in range(0, number_of_objects):
            s = Shape(width, height, relevant_colors)
            self.shapes.append(s)

    def save(self, name, width, height, border_width):
        """Save shapes to a file called *name* in SVG format.

        :param name: Name of SVG file to create.
        :param width: Width of source image.
        :param height: Height of source image.
        :param border_width: Width of black border for the objects.
        """
        with open(name, 'w') as f:
            f.write('<svg width="{}" height="{}">\n'.format(width, height))
            for i in range(0, len(self.shapes)):
                c = self.shapes[i].color
                x = self.shapes[i].x
                y = self.shapes[i].y
                w = self.shapes[i].width
                h = self.shapes[i].height
                if self.shapes[i].type == 0:
                    f.write('<rect x="{}" y="{}" width="{}" height="{}"'.format(x, y, w, h))
                else:
                    f.write('<circle cx="{}" cy="{}" r="{}"'.format(x, y, w//2))
                f.write(' style="fill:rgb{};stroke:black;stroke-width:{}"/>\n'.format(c, border_width))
            f.write('</svg>')

    def mutate_color(individual, relevant_colors, object_color_mutation_prob):
        """Color mutation: For each objects it is executed with probability
        *object_color_mutation_prob*. New color will be chosen randomly from
        the *relevant_colors* list.

        :param individual: An individual to be mutated.
        :param relevant_colors: List of relevant colors according to frequency.
        :param object_color_mutation_prob: Probability for each object to be
                                           mutated.
        """
        for index in range(len(individual.shapes)):
            if random.random() < object_color_mutation_prob:
                    individual.shapes[index].color = relevant_colors[random.randint(0, len(relevant_colors)-1)]

    def mutate_dimensions(individual, object_dimensions_mutation_prob):
        """
        Dimensions mutation: For each objects it is executed with probability
        *object_color_mutation_prob*. Dimensions will be changed via gaussian
        mutation.

        :param individual: An individual to be mutated.
        :param object_dimensions_mutation_prob: Probability for each object to
                                               be mutated.
        """
        for index in range(len(individual.shapes)):
            if random.random() < object_dimensions_mutation_prob:
                standard_deviation = 100
                original = individual.shapes[index].width
                individual.shapes[index].width = original + math.floor(random.gauss(0, standard_deviation))
                original = individual.shapes[index].height
                individual.shapes[index].height = original + math.floor(random.gauss(0, standard_deviation))

    def mutate_position(individual, object_position_mutation_prob):
        """
        Position mutation: For each objects it is executed with probability
        *object_position_mutation_prob*, position will be changed via
        gaussian mutation.

        :param individual: An individual to be mutated.
        :param object_position_mutation_prob: Probability for each object to be
                                              mutated.
        """
        for index in range(len(individual.shapes)):
            if random.random() < object_position_mutation_prob:
                original = individual.shapes[index].x
                individual.shapes[index].x = original + math.floor(random.gauss(0, 100))
                original = individual.shapes[index].y
                individual.shapes[index].y = original + math.floor(random.gauss(0, 100))

    def mutate_add(individual, relevant_colors, width, height):
        """
        Add mutation: New object is added.

        :param individual: An individual to be mutated.
        :param relevant_colors: List of relevant colors according to frequency.
        :param width: Width of source image.
        :param height: Height of source image.
        """
        s = Shape(width, height, relevant_colors)
        individual.shapes.append(s)

    def mutate_delete(individual):
        """
        Randomly chosen object is deleted. Object is deleted only if there are
        more than two objects left.

        :param individual: An individual to be mutated.
        """

        if len(individual.shapes) > 2:
            index = random.randint(0, len(individual.shapes) - 1)
            del individual.shapes[index]

    def evaluate_shapes_rgb(self, width, height, pixels, step_size):
        """Evaluate an individual acording to rgb based fitness. For each item in
        *pixels* calculate the difference with color value of corresponding point
        in an individual. The difference is calculated for red, green and blue
        part.

        :param self: An individual we want to evaluate.
        :param width: Width of source image.
        :param height: Height of source image.
        :param pixels: List of color values of pixels in source image.
        :param step_size: Indicates which points are evaluated and which
                          are skipped.
        :returns: Fitness value.
        """
        fitness = 0
        index = 0
        for i in range(0, width, step_size):
            for j in range(0, height, step_size):
                fitness_add = 3*256*256
                for s in range(len(self.shapes)):
                    if point_in_shape(self.shapes[s], i, j):
                        c1 = self.shapes[s].color
                        c2 = pixels[index]
                        fitness_add = (abs(c1[0] - c2[0]) * abs(c1[0] - c2[0]) +
                                       abs(c1[1] - c2[1]) * abs(c1[1] - c2[1]) +
                                       abs(c1[2] - c2[2]) * abs(c1[2] - c2[2]))
                index += 1
                fitness += fitness_add
        return fitness

    def evaluate(self, width, height, pixels, step_size):
        """Return tuple of fitness values. First is the rgb based fitness.
        Second is a number of objects in an individual.

        :param self: An individual we want to evaluate.
        :param width: Width of source image.
        :param height: Height of source image.
        :param pixels: List of color values of pixels in source image.
        :param step_size: Indicates which points are evaluated and which
                          are skipped for evaluate_shapes_rgb.
        :returns: Tuple of fitness values.
        """
        rgb_fitness = self.evaluate_shapes_rgb(width, height, pixels, step_size)
        count_fitness = len(self.shapes)
        return rgb_fitness, count_fitness


class Shape(object):
    """Class Shape represents an object that can be either a circle or
    a rectangle.
    """

    def __init__(self, width, height, relevant_colors):
        """Create object Shape. Set
        value for *self_color* from list of *relevant_colors*. Coordinates *x*
        and *y* are set as well as *width* and *height*. For circles, *width*
        stands for the diameter and *height* does not have any meaning. Value
        *type* is chosen randomly from (0,1) and indicates whether shape is
        rectangle(0) or circle(1).

        :param width: Width of source image.
        :param height: Height of source image.
        :param relevant_colors: List of relevant colors according to frequency.

        """
        self.color = relevant_colors[random.randint(0, len(relevant_colors)-1)]
        self.x = random.randint(0, width)
        self.y = random.randint(0, height)
        self.width = random.randint(0, width//2)
        self.height = random.randint(0, height//2)
        self.type = random.randint(0, 1)


def point_in_shape(shape, i, j):
    """Return whether point with coordinates *i* and *j* lies in a object of
    given shape or not.

    :param shape: Rectangle or circle.
    :param i: X coordinate of the point.
    :param j: Y coordinate of the point.
    :returns: True/False.
    """
    if shape.type == 0:
        if (i >= shape.x) and (i <= shape.width + shape.x):
            if (j >= shape.y) and (j <= shape.height + shape.y):
                return True
    else:
        length = math.sqrt(abs(shape.x - i)**2 + abs(shape.y - j)**2)
        if length <= (shape.width//2):
                return True
    return False


def choose_unique(hof):
    """Return unique items from Pareto front. Method coppies all items from
    Pareto front *hof* to a list. Items were sorted. Method goes through the
    list and if two consecutive values are identical, one of them is deleted.

    :param hof: Hall of fame - Pareto front.
    :returns: List of unique items in Pareto front.
    """
    unique = []
    for i in range(len(hof)):
        unique.append(hof[i])
    i = 0
    while i < len(unique)-1:
        if (unique[i].fitness.values[0] == unique[i+1].fitness.values[0]) and \
           (unique[i].fitness.values[1] == unique[i+1].fitness.values[1]):
            del unique[i+1]
            i -= 1
        i += 1
    return unique


def log_fitness(fitness_tuple, width, height, step_size):
    """Calculate normalized fitness for the first value in fitness tuple.

    :param fitness_tuple: Tuple of fitness values.
    :param width: Width of source image.
    :param height: Height of source image.
    :param step_size: Indicates which points are evaluated and which
                      are skipped.
    :returns: Normalized fitness.
    """
    for_one = fitness_tuple[0] / ((width // step_size) * (height // step_size))
    return math.sqrt(for_one)


def evolution(width, height, relevant_colors, pixels, number_of_objects, population_size, number_of_generations,
              color_mutation_prob, object_color_mutation_prob, dimensions_mutation_prob,
              object_dimensions_mutation_prob, position_mutation_prob, object_position_mutation_prob, add_prob,
              delete_prob, xover_prob, border_width):
    """Execute evolution. Fitness function is registered. Population of
    *population_size* individuals is created. Tools are registred into toolbox.
    Hall of fame method  - Pareto front - is implemented.
    For each generation: Offspring are chosen via TournamentDCD selection. Then
    crossover and mutations are executed and fitness is evaluated. New
    population is selected from individuals in last population and offspring.
    Hall of fame is updated and fitness is saved to log.
    After all generations, unique individuals from Pareto front are chosen and
    saved as SVG files. Log of their fitnesses is created.

    :param width: Width of source image.
    :param height: Height of source image.
    :param relevant_colors: List of relevant colors according to frequency.
    :param pixels: List of color values of pixels in source image.
    :param number_of_objects: Initial number of shapes in one individual.
    :param population_size: Number of individuals in a population.
    :param number_of_generations: Total number of generations to be evolved.
    :param color_mutation_prob: Probability that color mutation will be done.
    :param object_color_mutation_prob: Probability that color mutation will be
                                       executed for an object.
    :param dimensions_mutation_prob: Probability that dimensions mutation will be
                                    done.
    :param object_dimensions_mutation_prob: Probability that dimensions mutation
                                           will be executed for an object.
    :param position_mutation_prob: Probability that position mutation will be
                                   done.
    :param object_position_mutation_prob: Probability that position mutation
                                           will be executed for an object.
    :param add_prob: Probability that add_object mutation will be done.
    :param delete_prob: Probability that delete_object mutation will be done.
    :param xover_prob: Probability that crossover will be done.
    """

    filename = "multi-objective_results"
    os.mkdir(filename)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", Picture, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("individual", creator.Individual, width=width, height=height, relevant_colors=relevant_colors,
                     number_of_objects=number_of_objects)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mutate_color", Picture.mutate_color, relevant_colors=relevant_colors,
                     object_color_mutation_prob=object_color_mutation_prob)
    toolbox.register("mutate_dimensions", Picture.mutate_dimensions,
                     object_dimensions_mutation_prob=object_dimensions_mutation_prob)
    toolbox.register("mutate_position", Picture.mutate_position,
                     object_position_mutation_prob=object_position_mutation_prob)
    toolbox.register("add_object", Picture.mutate_add, relevant_colors=relevant_colors, width=width, height=height)
    toolbox.register("delete_object", Picture.mutate_delete)
    toolbox.register("mate", tools.cxOnePoint)

    toolbox.register("evaluate", Picture.evaluate, width=width, height=height, pixels=pixels, step_size=step_size)
    toolbox.register("select", tools.selNSGA2)

    population = toolbox.population(n=population_size)
    fitnesses = map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    population = toolbox.select(population, len(population))
    hof = tools.ParetoFront()
    hof.update(population)
    for k in range(number_of_generations):
        offspring = tools.selTournamentDCD(population, len(population))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for item1, item2 in zip(offspring[1::2], offspring[::2]):
            if random.random() < xover_prob:
                toolbox.mate(item1.shapes, item2.shapes)
                del item1.fitness.values
                del item2.fitness.values

        for mutant in offspring:
            if random.random() < color_mutation_prob:
                toolbox.mutate_color(mutant)
                del mutant.fitness.values

        for mutant in offspring:
            if random.random() < dimensions_mutation_prob:
                toolbox.mutate_dimensions(mutant)
                del mutant.fitness.values

        for mutant in offspring:
            if random.random() < position_mutation_prob:
                toolbox.mutate_position(mutant)
                del mutant.fitness.values

        for mutant in offspring:
            if random.random() < add_prob:
                toolbox.add_object(mutant)
                del mutant.fitness.values

        for mutant in offspring:
            if random.random() < delete_prob:
                toolbox.delete_object(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population = toolbox.select(population + offspring, population_size)
        hof.update(population)
    unique = choose_unique(hof)
    for j in range(len(unique)):
        unique[j].save(filename + "/" + str(j) + ".svg", width, height, border_width)
        print(log_fitness(unique[j].fitness.values, width, height, step_size), unique[j].fitness.values[1])


if __name__ == '__main__':
    random.seed("seed")
    configuration = json.load(open(sys.argv[1]))
    image_name = configuration['image_name']
    step_size = configuration['step_size']
    population_size = configuration['population_size']
    number_of_relevant_colors = configuration['number_of_relevant_colors']
    color_limitation_size = configuration['color_limitation_size']
    number_of_generations = configuration['number_of_generations']
    number_of_objects = configuration['number_of_objects']

    mutation_parameters = configuration['operators']['mutation']
    color_mutation_prob = mutation_parameters['color_mutation_prob']
    object_color_mutation_prob = mutation_parameters['object_color_mutation_prob']
    dimensions_mutation_prob = mutation_parameters['dimensions_mutation_prob']
    object_dimensions_mutation_prob = mutation_parameters['object_dimensions_mutation_prob']
    position_mutation_prob = mutation_parameters['position_mutation_prob']
    object_position_mutation_prob = mutation_parameters['object_position_mutation_prob']
    add_prob = mutation_parameters['add_prob']
    delete_prob = mutation_parameters['delete_prob']
    xover_prob = configuration['operators']['xover_prob']

    image_parameters = configuration['image_properties']
    border_width = image_parameters['border_width']

    image = MyImage(image_name, color_limitation_size, number_of_relevant_colors, step_size)
    evolution(image.width, image.height, image.relevant_colors, image.pixels, number_of_objects, population_size,
              number_of_generations, color_mutation_prob, object_color_mutation_prob, dimensions_mutation_prob,
              object_dimensions_mutation_prob, position_mutation_prob, object_position_mutation_prob, add_prob,
              delete_prob, xover_prob, border_width)
