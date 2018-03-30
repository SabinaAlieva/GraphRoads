# GraphRoads
Для получения результатов необходимо запустить скрипт run.ipynb.
В скрипте есть поясняющие комментарии. К сожалению, исходный файл xml весил 56 Мб и не загрузился. Матрица смежности не получилась из-за большого размера, но описан алгоритм ее построения.

Для парсинга xml используется библиотека lxml, etree, objectify. 
Отбираем дороги из тегов way, в которых указан tag с атрибутом k = highway. 
Затем создается словарь для быстрого получения по номеру узла его координат.
Создаем словарь, в котором номеру узла соответствует количество его вхождений в дороги.
Создаем словарь, в котором номеру узла соответствует количество ребер ему инцидентных. (Ребер в одной дороге n - 1, где n - количество узлов в дороге)
Удалить ненобходимо те узлы, которые имеют степень 2 и встречаются в 1 дороге или не встречаются вообще ни в одной дороге.
Для удобства вся отрисовка графиков закомментирована. 
Создаем словарь списка смежности по принципу: если текущий узел в дороге не необходим к удалению то его и следующий узел записываем в список смежности по соответствующему ключу, так как два этих узла будут образовывать ребро. (если следующий тоже не необходим к удалению, и т д)
Убираем из списка смежности повторяющиеся значения вершин. 
В csv файл списка смежности записываем словарь списка смежности. (В первом столбце id вершины)
В матрицу смежности записываем построчно массив из нулей, в котором на месте ребра для данного узла стоят 1.

В результате получены: 
test_sochi.png - исходная отрисовка дорог, без удаления лишиних узлов.
test_sochi2.png - то же самое в большом разрешении.
testo_sochi.jpg - отрисовка по веришинам, которые остались после удаления.
mini-sochi_map2.jpg - скриншот исходной карты для сравнения.
