; tamp/pddl/fetch_stream.pddl -- streams for fetch_domain.pddl.  Implementations (and the checker behind each): tamp/streams.py.
(define (stream yor-fetch)
  (:stream s-dock
    :inputs (?s)
    :domain (Surface ?s)
    :outputs (?q)
    :certified (and (Conf ?q) (Dock ?q ?s)))

  (:stream s-grasp
    :inputs (?o)
    :domain (Item ?o)
    :outputs (?g)
    :certified (Grasp ?o ?g))

  (:stream s-place
    :inputs (?o ?s)
    :domain (and (Item ?o) (Surface ?s))
    :outputs (?p)
    :certified (and (Pose ?o ?p) (Supported ?o ?p ?s)))

  (:stream t-reach
    :inputs (?o ?p ?g ?q)
    :domain (and (Pose ?o ?p) (Grasp ?o ?g) (Conf ?q))
    :certified (Reach ?o ?p ?g ?q))

  (:stream s-route-free
    :inputs (?q1 ?q2)
    :domain (and (Conf ?q1) (Conf ?q2))
    :outputs (?r)
    :certified (FreeRoute ?q1 ?q2 ?r))

  (:stream s-route-hold
    :inputs (?o ?g ?q1 ?q2)
    :domain (and (Grasp ?o ?g) (Conf ?q1) (Conf ?q2))
    :outputs (?r)
    :certified (HoldRoute ?o ?g ?q1 ?q2 ?r))
)
