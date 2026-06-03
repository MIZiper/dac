from dac.core import Container, DataNode, DataContext


def test_qualified_name_children_avoid_collisions():
    ctx = DataContext(Container())
    parent_a = DataNode(name="parent_a")
    parent_b = DataNode(name="parent_b")
    child_a = DataNode(name="chan")
    child_b = DataNode(name="chan")
    parent_a.add_child(child_a)
    parent_b.add_child(child_b)
    ctx.add_node(parent_a)
    ctx.add_node(parent_b)

    node_a = ctx.get_node_of_type("parent_a/chan", DataNode)
    node_b = ctx.get_node_of_type("parent_b/chan", DataNode)
    assert node_a is child_a
    assert node_b is child_b
    assert node_a is not node_b


def test_qualified_name_flat_lookup_returns_none_for_children():
    ctx = DataContext(Container())
    parent = DataNode(name="parent")
    child = DataNode(name="child")
    parent.add_child(child)
    ctx.add_node(parent)

    assert ctx.get_node_of_type("parent", DataNode) is parent
    assert ctx.get_node_of_type("parent/child", DataNode) is child
    assert ctx.get_node_of_type("child", DataNode) is None


def test_qualified_name_top_level_unchanged():
    ctx = DataContext(Container())
    node = DataNode(name="top_level")
    ctx.add_node(node)

    assert ctx.get_node_of_type("top_level", DataNode) is node


def test_qualified_name_uuid_lookup_still_works():
    ctx = DataContext(Container())
    parent = DataNode(name="parent")
    child = DataNode(name="child")
    parent.add_child(child)
    ctx.add_node(parent)

    assert ctx.get_node_by_uuid(child.uuid) is child
    assert ctx.get_node_by_uuid(parent.uuid) is parent


def test_qualified_name_rename_cascades_positively():
    ctx = DataContext(Container())
    parent = DataNode(name="old_parent")
    child = DataNode(name="child")
    parent.add_child(child)
    ctx.add_node(parent)

    ctx.rename_node_to(parent, "new_parent")

    assert ctx.get_node_of_type("new_parent", DataNode) is parent
    assert ctx.get_node_of_type("new_parent/child", DataNode) is child


def test_qualified_name_rename_cascades_negatively():
    ctx = DataContext(Container())
    parent = DataNode(name="old_parent")
    child = DataNode(name="child")
    parent.add_child(child)
    ctx.add_node(parent)

    ctx.rename_node_to(parent, "new_parent")

    assert ctx.get_node_of_type("old_parent", DataNode) is None
    assert ctx.get_node_of_type("old_parent/child", DataNode) is None


def test_qualified_name_deeply_nested():
    ctx = DataContext(Container())
    root = DataNode(name="root")
    mid = DataNode(name="mid")
    leaf = DataNode(name="leaf")
    mid.add_child(leaf)
    root.add_child(mid)
    ctx.add_node(root)

    assert ctx.get_node_of_type("root", DataNode) is root
    assert ctx.get_node_of_type("root/mid", DataNode) is mid
    assert ctx.get_node_of_type("root/mid/leaf", DataNode) is leaf


def test_qualified_name_sibling_collision_deep():
    ctx = DataContext(Container())
    root1 = DataNode(name="root1")
    root2 = DataNode(name="root2")
    child1 = DataNode(name="child")
    child2 = DataNode(name="child")
    grand1 = DataNode(name="grand")
    grand2 = DataNode(name="grand")
    child1.add_child(grand1)
    child2.add_child(grand2)
    root1.add_child(child1)
    root2.add_child(child2)
    ctx.add_node(root1)
    ctx.add_node(root2)

    assert ctx.get_node_of_type("root1/child/grand", DataNode) is grand1
    assert ctx.get_node_of_type("root2/child/grand", DataNode) is grand2
    a = ctx.get_node_of_type("root1/child/grand", DataNode)
    b = ctx.get_node_of_type("root2/child/grand", DataNode)
    assert a is not b


def test_qualified_name_rename_deep_cascades():
    ctx = DataContext(Container())
    root = DataNode(name="old_root")
    mid = DataNode(name="mid")
    leaf = DataNode(name="leaf")
    mid.add_child(leaf)
    root.add_child(mid)
    ctx.add_node(root)

    ctx.rename_node_to(root, "new_root")

    assert ctx.get_node_of_type("new_root/mid/leaf", DataNode) is leaf
    assert ctx.get_node_of_type("old_root/mid/leaf", DataNode) is None


def test_qualified_name_rename_mid_level():
    ctx = DataContext(Container())
    root = DataNode(name="root")
    mid = DataNode(name="old_mid")
    leaf = DataNode(name="leaf")
    mid.add_child(leaf)
    root.add_child(mid)
    ctx.add_node(root)

    ctx.rename_node_to(mid, "new_mid")

    assert ctx.get_node_of_type("root/new_mid/leaf", DataNode) is leaf
    assert ctx.get_node_of_type("root/old_mid/leaf", DataNode) is None
