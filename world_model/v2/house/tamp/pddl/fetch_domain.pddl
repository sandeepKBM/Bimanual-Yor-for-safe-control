; tamp/pddl/fetch_domain.pddl -- YOR mobile bimanual fetch / carry / place.  SIMULATION ONLY.
; Static predicates are produced by STREAMS (fetch_stream.pddl); each stream output is admitted by one of our checkers (see tamp/streams.py):
;   Dock       <- C2 static: the rest state at the dock passes the static clearance row (dock buffer) under the pose box
;   Grasp      <- C3: the grasp holds for EVERY corner of the item's (mass, CoM, mu, accel) box, and the executor has a skill for it
;   Supported  <- support test: the item footprint lies inside the surface top
;   Reach      <- kinematic test: from dock conf q the hold pose of the item matches placement p (within the align-jog range)
;   FreeRoute / HoldRoute <- C1+C2 static route certificate (T3 H0): from rest at every sample pose along the route, a forward candidate
;                            (nominal + cup-safe backup) is admitted with NO people; HoldRoute also carries the payload (cup) rows.
; People are NOT certified at plan time: the 50 Hz gatekeeper certifies every executed segment against the live set-valued belief (T1).
(define (domain yor-fetch)
  (:requirements :strips :equality)
  (:predicates
    ; static / stream-certified
    (Item ?o)
    (Surface ?s)
    (Conf ?q)
    (Dock ?q ?s)
    (Pose ?o ?p)
    (Supported ?o ?p ?s)
    (Grasp ?o ?g)
    (Reach ?o ?p ?g ?q)
    (FreeRoute ?q1 ?q2 ?r)
    (HoldRoute ?o ?g ?q1 ?q2 ?r)
    ; fluents
    (AtConf ?q)
    (AtPose ?o ?p)
    (Holding ?o ?g)
    (HandsEmpty)
    (CanMove)
    ; derived
    (On ?o ?s)
    (HoldingAny ?o)
  )

  (:derived (On ?o ?s)
    (exists (?p) (and (Supported ?o ?p ?s) (AtPose ?o ?p))))

  (:derived (HoldingAny ?o)
    (exists (?g) (and (Grasp ?o ?g) (Holding ?o ?g))))

  (:action move_free
    :parameters (?q1 ?q2 ?r)
    :precondition (and (AtConf ?q1) (HandsEmpty) (CanMove) (FreeRoute ?q1 ?q2 ?r))
    :effect (and (AtConf ?q2) (not (AtConf ?q1)) (not (CanMove))))

  (:action move_holding
    :parameters (?o ?g ?q1 ?q2 ?r)
    :precondition (and (AtConf ?q1) (Holding ?o ?g) (CanMove) (HoldRoute ?o ?g ?q1 ?q2 ?r))
    :effect (and (AtConf ?q2) (not (AtConf ?q1)) (not (CanMove))))

  (:action pick
    :parameters (?o ?p ?g ?q)
    :precondition (and (AtConf ?q) (AtPose ?o ?p) (HandsEmpty) (Reach ?o ?p ?g ?q))
    :effect (and (Holding ?o ?g) (CanMove) (not (AtPose ?o ?p)) (not (HandsEmpty))))

  (:action place
    :parameters (?o ?p ?g ?q ?s)
    :precondition (and (AtConf ?q) (Holding ?o ?g) (Reach ?o ?p ?g ?q) (Supported ?o ?p ?s) (Dock ?q ?s))
    :effect (and (AtPose ?o ?p) (HandsEmpty) (CanMove) (not (Holding ?o ?g))))
)
