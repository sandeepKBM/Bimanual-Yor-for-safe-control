; Covered pan on a stove, two arms.  Author-correct PDDL: the lid is NOT part of the pan (it is a loose body).
(define (domain covered-pan)
  (:requirements :strips :typing :negative-preconditions :equality)
  (:types arm part)
  (:constants L R - arm handle lid rim - part)
  (:predicates (free ?a - arm) (holding ?a - arm ?p - part) (on-stove) (lifted) (overhang) (lid-secured)
               (can-grasp ?a - arm ?p - part) (handle-side ?a - arm) (far-side ?a - arm) (part-of-pan ?p - part))
  (:action pick
    :parameters (?a - arm ?p - part)
    :precondition (and (free ?a) (on-stove) (can-grasp ?a ?p))
    :effect (and (holding ?a ?p) (not (free ?a))))
  (:action push
    :parameters (?a - arm)
    :precondition (and (holding ?a handle) (handle-side ?a) (on-stove) (not (overhang)))
    :effect (overhang))
  (:action clamp
    :parameters (?a - arm)
    :precondition (and (free ?a) (far-side ?a) (overhang) (on-stove))
    :effect (and (holding ?a rim) (lid-secured) (not (free ?a))))
  (:action lift-one
    :parameters (?a - arm ?p - part ?b - arm)
    :precondition (and (holding ?a ?p) (part-of-pan ?p) (on-stove) (free ?b) (not (= ?a ?b)))
    :effect (and (lifted) (not (on-stove))))
  (:action lift-two
    :parameters (?a1 - arm ?p1 - part ?a2 - arm ?p2 - part)
    :precondition (and (holding ?a1 ?p1) (holding ?a2 ?p2) (part-of-pan ?p1) (part-of-pan ?p2) (on-stove) (not (= ?a1 ?a2)))
    :effect (and (lifted) (not (on-stove)))))

(define (problem lift-covered-pan)
  (:domain covered-pan)
  (:init (free L) (free R) (on-stove)
         (can-grasp L handle) (can-grasp R handle) (can-grasp L lid) (can-grasp R lid)
         (handle-side L) (far-side R) (part-of-pan handle) (part-of-pan rim))
  (:goal (lifted)))
